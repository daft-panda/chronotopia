// LocationService.kt
package com.example.trip_tracker

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ServiceInfo
import android.location.Geocoder
import android.location.Location
import android.os.BatteryManager
import android.os.Build
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.telephony.TelephonyManager
import androidx.core.app.NotificationCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel
import org.json.JSONObject
import java.util.Locale
import java.util.concurrent.TimeUnit
import java.util.Date
import java.text.SimpleDateFormat
import java.io.File
import java.io.FileWriter

class LocationService : Service() {
    private var wakeLock: PowerManager.WakeLock? = null
    private var isServiceRunning = false
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private lateinit var locationCallback: LocationCallback
    
    // For reverse geocoding
    private lateinit var geocoder: Geocoder
    
    // For battery monitoring
    private var batteryLevel: Int = -1
    private val batteryReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
            val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
            batteryLevel = (level * 100 / scale.toFloat()).toInt()
        }
    }
    
    companion object {
        private const val NOTIFICATION_ID = 12345
        private const val CHANNEL_ID = "location_service_channel"
        
        private const val LOCATION_INTERVAL_ACTIVE = 10000L  // 10 seconds when app is active
        private const val LOCATION_INTERVAL_BACKGROUND = 180000L  // 3 minutes in background
        private const val LOCATION_FASTEST_INTERVAL = 5000L  // 5 seconds
        private const val LOCATION_SMALLEST_DISPLACEMENT = 10f  // 10 meters
        
        private var eventSink: EventChannel.EventSink? = null
        
        fun setEventSink(sink: EventChannel.EventSink?) {
            eventSink = sink
        }
        
        fun registerWith(methodChannel: MethodChannel, eventChannel: EventChannel) {
            methodChannel.setMethodCallHandler { call, result ->
                when (call.method) {
                    "initialize" -> {
                        result.success(null)
                    }
                    "startTracking" -> {
                        val intent = Intent(methodChannel.plugin.context, LocationService::class.java)
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                            methodChannel.plugin.context.startForegroundService(intent)
                        } else {
                            methodChannel.plugin.context.startService(intent)
                        }
                        result.success(null)
                    }
                    "stopTracking" -> {
                        val intent = Intent(methodChannel.plugin.context, LocationService::class.java)
                        methodChannel.plugin.context.stopService(intent)
                        result.success(null)
                    }
                    else -> result.notImplemented()
                }
            }
            
            eventChannel.setStreamHandler(object : EventChannel.StreamHandler {
                override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
                    eventSink = events
                }
                
                override fun onCancel(arguments: Any?) {
                    eventSink = null
                }
            })
        }
    }
    
    override fun onCreate() {
        super.onCreate()
        
        // Initialize location client
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        
        // Initialize geocoder
        geocoder = Geocoder(this, Locale.getDefault())
        
        // Register battery receiver
        registerReceiver(batteryReceiver, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        
        // Create location callback
        locationCallback = object : LocationCallback() {
            override fun onLocationResult(locationResult: LocationResult) {
                super.onLocationResult(locationResult)
                for (location in locationResult.locations) {
                    processLocation(location)
                }
            }
        }
    }
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground()
        startLocationUpdates()
        
        // Acquire wake lock to keep CPU running
        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "TripTracker::LocationWakeLock"
        )
        wakeLock?.acquire(10*60*1000L /*10 minutes*/)
        
        isServiceRunning = true
        
        // If service gets killed, restart it
        return START_STICKY
    }
    
    override fun onBind(intent: Intent?): IBinder? {
        return null
    }
    
    override fun onDestroy() {
        super.onDestroy()
        
        stopLocationUpdates()
        
        try {
            unregisterReceiver(batteryReceiver)
        } catch (e: Exception) {
            // Receiver might not be registered
        }
        
        wakeLock?.let {
            if (it.isHeld) {
                it.release()
            }
        }
        
        isServiceRunning = false
    }
    
    private fun startForeground() {
        // Create notification channel for Android O and above
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Location Service Channel",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Used for background location tracking"
                setShowBadge(false)
            }
            
            val notificationManager = getSystemService(NotificationManager::class.java)
            notificationManager.createNotificationChannel(channel)
        }
        
        // Create notification for foreground service
        val notificationIntent = packageManager.getLaunchIntentForPackage(packageName)
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            notificationIntent,
            PendingIntent.FLAG_IMMUTABLE
        )
        
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Trip Tracker Active")
            .setContentText("Tracking your location and activities")
            .setSmallIcon(R.drawable.ic_notification) // Make sure this exists in your resources
            .setContentIntent(pendingIntent)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
        
        // Start foreground service with notification
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }
    
    private fun startLocationUpdates() {
        val locationRequest = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, LOCATION_INTERVAL_ACTIVE)
            .setMinUpdateIntervalMillis(LOCATION_FASTEST_INTERVAL)
            .setMinUpdateDistanceMeters(LOCATION_SMALLEST_DISPLACEMENT)
            .build()
        
        try {
            fusedLocationClient.requestLocationUpdates(
                locationRequest,
                locationCallback,
                Looper.getMainLooper()
            )
        } catch (e: SecurityException) {
            // Handle permission issues
            e.printStackTrace()
        }
    }
    
    private fun stopLocationUpdates() {
        fusedLocationClient.removeLocationUpdates(locationCallback)
    }
    
    private fun processLocation(location: Location) {
        // Get network type
        val telephonyManager = getSystemService(Context.TELEPHONY_SERVICE) as TelephonyManager
        val networkType = when (telephonyManager.dataNetworkType) {
            TelephonyManager.NETWORK_TYPE_GPRS,
            TelephonyManager.NETWORK_TYPE_EDGE,
            TelephonyManager.NETWORK_TYPE_CDMA,
            TelephonyManager.NETWORK_TYPE_1xRTT,
            TelephonyManager.NETWORK_TYPE_IDEN,
            TelephonyManager.NETWORK_TYPE_GSM -> "2G"
            
            TelephonyManager.NETWORK_TYPE_UMTS,
            TelephonyManager.NETWORK_TYPE_EVDO_0,
            TelephonyManager.NETWORK_TYPE_EVDO_A,
            TelephonyManager.NETWORK_TYPE_HSDPA,
            TelephonyManager.NETWORK_TYPE_HSUPA,
            TelephonyManager.NETWORK_TYPE_HSPA,
            TelephonyManager.NETWORK_TYPE_EVDO_B,
            TelephonyManager.NETWORK_TYPE_EHRPD,
            TelephonyManager.NETWORK_TYPE_HSPAP,
            TelephonyManager.NETWORK_TYPE_TD_SCDMA -> "3G"
            
            TelephonyManager.NETWORK_TYPE_LTE,
            TelephonyManager.NETWORK_TYPE_IWLAN,
            TelephonyManager.NETWORK_TYPE_LTE_CA -> "4G"
            
            TelephonyManager.NETWORK_TYPE_NR -> "5G"
            
            else -> "Unknown"
        }
        
        // Send the location data to Flutter - geocoding will be done on the backend
        sendLocationData(location, networkType)
    }
    
    private fun sendLocationData(
        location: Location,
        networkType: String
    ) {
        // Create JSON object with all available data
        val locationData = JSONObject().apply {
            put("latitude", location.latitude)
            put("longitude", location.longitude)
            put("altitude", if (location.hasAltitude()) location.altitude else null)
            put("accuracy", if (location.hasAccuracy()) location.accuracy else null)
            put("heading", if (location.hasBearing()) location.bearing else null)
            put("speed", if (location.hasSpeed()) location.speed else null)
            put("timestamp", SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US)
                .apply { timeZone = java.util.TimeZone.getTimeZone("UTC") }
                .format(Date(location.time)))
            put("batteryLevel", batteryLevel)
            put("networkType", networkType)
            
            // Raw sensor data when available
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                put("verticalAccuracy", if (location.hasVerticalAccuracy()) location.verticalAccuracyMeters else null)
                put("speedAccuracy", if (location.hasSpeedAccuracy()) location.speedAccuracyMetersPerSecond else null)
                put("bearingAccuracy", if (location.hasBearingAccuracy()) location.bearingAccuracyDegrees else null)
            }
            
            // Include the provider information
            put("provider", location.provider)
            
            // Add mock location flag for debugging/testing
            put("isMockLocation", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN_MR2) {
                location.isFromMockProvider
            } else {
                false
            })
        }
        
        // Cache the location data and attempt to send via gRPC
        grpcClient.cacheLocationData(locationData)
        
        // Also send to Flutter via event channel if available (for UI updates when app is in foreground)
        eventSink?.success(locationData.toString())
    }.verticalAccuracyMeters else null)
                put("speedAccuracy", if (location.hasSpeedAccuracy()) location.speedAccuracyMetersPerSecond else null)
                put("bearingAccuracy", if (location.hasBearingAccuracy()) location.bearingAccuracyDegrees else null)
            }
            
            // Include the provider information
            put("provider", location.provider)
            
            // Add mock location flag for debugging/testing
            put("isMockLocation", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN_MR2) {
                location.isFromMockProvider
            } else {
                false
            })
        }
        
        // Cache the location data
        grpcClient.cacheLocationData(locationData)
        
        // Also send to Flutter via event channel if available (for UI updates when app is in foreground)
        eventSink?.success(locationData.toString())
    }.verticalAccuracyMeters else null)
                put("speedAccuracy", if (location.hasSpeedAccuracy()) location.speedAccuracyMetersPerSecond else null)
                put("bearingAccuracy", if (location.hasBearingAccuracy()) location.bearingAccuracyDegrees else null)
            }
            
            // Include the provider information
            put("provider", location.provider)
            
            // Add mock location flag for debugging/testing
            put("isMockLocation", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN_MR2) {
                location.isFromMockProvider
            } else {
                false
            })
        }
        
        // Send the data to Flutter via event channel
        eventSink?.success(locationData.toString())
        
        // Also save to local storage as backup
        saveLocationToStorage(locationData.toString())
    }
    
    private fun saveLocationToStorage(locationData: String) {
        try {
            val dir = File(applicationContext.filesDir, "location_data")
            if (!dir.exists()) {
                dir.mkdirs()
            }
            
            val timestamp = System.currentTimeMillis()
            val file = File(dir, "location_$timestamp.json")
            
            FileWriter(file).use { writer ->
                writer.write(locationData)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}