// ActivityRecognition.kt
package com.example.trip_tracker

import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import com.google.android.gms.location.ActivityRecognition
import com.google.android.gms.location.ActivityRecognitionClient
import com.google.android.gms.location.ActivityRecognitionResult
import com.google.android.gms.location.DetectedActivity
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel
import org.json.JSONObject
import kotlin.math.abs

class ActivityRecognitionService : Service() {
    private var wakeLock: PowerManager.WakeLock? = null
    private var isServiceRunning = false
    private lateinit var activityRecognitionClient: ActivityRecognitionClient
    private lateinit var activityRecognitionPendingIntent: PendingIntent
    
    // gRPC client for sending data
    private lateinit var grpcClient: GrpcClient
    
    // Step counter
    private var lastStepCount: Long = 0
    private var lastStepTimestamp: Long = 0
    
    // Activity data
    private var currentActivity: Int = DetectedActivity.UNKNOWN
    private var currentConfidence: Int = 0
    private var activityStartTime: Long = 0
    
    companion object {
        private const val TAG = "ActivityRecognition"
        private const val ACTIVITY_DETECTION_INTERVAL = 10000L // 10 seconds
        private const val ACTION_ACTIVITY_RECOGNITION = "com.example.trip_tracker.ACTION_ACTIVITY_RECOGNITION"
        
        private var eventSink: EventChannel.EventSink? = null
        
        fun setEventSink(sink: EventChannel.EventSink?) {
            eventSink = sink
        }
        
        fun registerWith(methodChannel: MethodChannel, eventChannel: EventChannel) {
            methodChannel.setMethodCallHandler { call, result ->
                when (call.method) {
                    "initialize" -> {
                        result.success(true)
                    }
                    "startTracking" -> {
                        val intent = Intent(methodChannel.plugin.context, ActivityRecognitionService::class.java)
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                            methodChannel.plugin.context.startForegroundService(intent)
                        } else {
                            methodChannel.plugin.context.startService(intent)
                        }
                        result.success(null)
                    }
                    "stopTracking" -> {
                        val intent = Intent(methodChannel.plugin.context, ActivityRecognitionService::class.java)
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
    
    private val activityRecognitionReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action == ACTION_ACTIVITY_RECOGNITION) {
                if (ActivityRecognitionResult.hasResult(intent)) {
                    val result = ActivityRecognitionResult.extractResult(intent)
                    handleActivityDetected(result)
                }
            }
        }
    }
    
    override fun onCreate() {
        super.onCreate()
        
        activityRecognitionClient = ActivityRecognition.getClient(this)
        
        val intent = Intent(ACTION_ACTIVITY_RECOGNITION)
        activityRecognitionPendingIntent = PendingIntent.getBroadcast(
            this,
            0,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_MUTABLE
        )
        
        registerReceiver(activityRecognitionReceiver, IntentFilter(ACTION_ACTIVITY_RECOGNITION))
        
        // Initialize gRPC client
        grpcClient = GrpcClient(applicationContext)
        
        // Initialize step counter
        lastStepCount = 0
        lastStepTimestamp = System.currentTimeMillis()
        activityStartTime = System.currentTimeMillis()
    }
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // If reusing notification from location service, get its notification
        val notification = (applicationContext.getSystemService(Context.NOTIFICATION_SERVICE) as android.app.NotificationManager)
            .activeNotifications
            .firstOrNull { it.id == LocationService.NOTIFICATION_ID }
            ?.notification
            
        // If notification exists, reuse it
        if (notification != null) {
            startForeground(LocationService.NOTIFICATION_ID, notification)
        }
        
        startActivityRecognition()
        
        // Acquire wake lock to keep CPU running
        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "TripTracker::ActivityWakeLock"
        )
        wakeLock?.acquire(10*60*1000L /*10 minutes*/)
        
        isServiceRunning = true
        
        return START_STICKY
    }
    
    override fun onBind(intent: Intent?): IBinder? {
        return null
    }
    
    override fun onDestroy() {
        super.onDestroy()
        
        stopActivityRecognition()
        
        try {
            unregisterReceiver(activityRecognitionReceiver)
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
    
    private fun startActivityRecognition() {
        try {
            activityRecognitionClient.requestActivityUpdates(
                ACTIVITY_DETECTION_INTERVAL,
                activityRecognitionPendingIntent
            )
        } catch (e: SecurityException) {
            Log.e(TAG, "No permission to request activity updates", e)
        }
    }
    
    private fun stopActivityRecognition() {
        try {
            activityRecognitionClient.removeActivityUpdates(activityRecognitionPendingIntent)
        } catch (e: Exception) {
            Log.e(TAG, "Error removing activity updates", e)
        }
    }
    
    private fun handleActivityDetected(result: ActivityRecognitionResult) {
        val mostProbableActivity = result.mostProbableActivity
        val activityType = mostProbableActivity.type
        val confidence = mostProbableActivity.confidence
        
        // If we detect a significant change in activity or confidence
        if (isSignificantActivityChange(activityType, confidence)) {
            // Send previous activity if it exists
            if (currentActivity != DetectedActivity.UNKNOWN && currentConfidence > 0) {
                sendActivityData(
                    activityType = currentActivity,
                    confidence = currentConfidence,
                    startTime = activityStartTime,
                    endTime = System.currentTimeMillis()
                )
            }
            
            // Update current activity data
            currentActivity = activityType
            currentConfidence = confidence
            activityStartTime = System.currentTimeMillis()
        }
        
        // Collect movement data
        collectMovementData(activityType)
    }
    
    private fun isSignificantActivityChange(newType: Int, newConfidence: Int): Boolean {
        // Consider it significant if activity type changes
        if (newType != currentActivity) {
            return true
        }
        
        // Or if confidence changes significantly (more than 20%)
        if (abs(newConfidence - currentConfidence) > 20) {
            return true
        }
        
        return false
    }
    
    private fun collectMovementData(activityType: Int) {
        // Calculate time elapsed since last update
        val currentTime = System.currentTimeMillis()
        val elapsedTimeMinutes = (currentTime - lastStepTimestamp) / 1000.0 / 60.0
        
        // Estimate movement data based on activity type and time elapsed
        var estimatedSteps = 0L
        var estimatedDistance = 0.0
        
        when (activityType) {
            DetectedActivity.WALKING -> {
                // ~100 steps per minute, ~0.8m per step
                estimatedSteps = (100 * elapsedTimeMinutes).toLong()
                estimatedDistance = estimatedSteps * 0.8
            }
            DetectedActivity.RUNNING -> {
                // ~150 steps per minute, ~1.0m per step
                estimatedSteps = (150 * elapsedTimeMinutes).toLong()
                estimatedDistance = estimatedSteps * 1.0
            }
            DetectedActivity.ON_BICYCLE -> {
                // Minimal steps, ~250m per minute
                estimatedSteps = (10 * elapsedTimeMinutes).toLong()
                estimatedDistance = 250.0 * elapsedTimeMinutes
            }
            DetectedActivity.IN_VEHICLE -> {
                // Minimal steps, variable distance (not reliable)
                estimatedSteps = (5 * elapsedTimeMinutes).toLong()
                estimatedDistance = 0.0 // Not reliable for vehicles
            }
            else -> {
                // Minimal values for other activities
                estimatedSteps = (10 * elapsedTimeMinutes).toLong()
                estimatedDistance = 10.0 * elapsedTimeMinutes
            }
        }
        
        // Update step count
        lastStepCount += estimatedSteps
        lastStepTimestamp = currentTime
        
        // Send the estimated movement data
        sendActivityData(
            activityType = activityType,
            confidence = currentConfidence,
            startTime = activityStartTime,
            endTime = currentTime,
            stepCount = estimatedSteps,
            distance = estimatedDistance
        )
    }
    
    private fun sendActivityData(
        activityType: Int,
        confidence: Int,
        startTime: Long,
        endTime: Long,
        stepCount: Long? = null,
        distance: Double? = null
    ) {
        // Create JSON object with activity data
        val activityData = JSONObject().apply {
            put("type", getActivityTypeString(activityType))
            put("confidence", confidence)
            put("timestamp", endTime)
            put("startTime", startTime)
            put("endTime", endTime)
            
            // Include movement data when available
            if (stepCount != null) put("stepCount", stepCount)
            if (distance != null) put("distance", distance)
        }
        
        // Cache and send via gRPC
        grpcClient.cacheActivityData(activityData)
        
        // Also send to Flutter via event channel if available (for UI updates when app is in foreground)
        eventSink?.success(activityData.toString())
    }
    
    private fun getActivityTypeString(activityType: Int): String {
        return when (activityType) {
            DetectedActivity.IN_VEHICLE -> "in_vehicle"
            DetectedActivity.ON_BICYCLE -> "on_bicycle"
            DetectedActivity.ON_FOOT -> "on_foot"
            DetectedActivity.RUNNING -> "running"
            DetectedActivity.STILL -> "still"
            DetectedActivity.TILTING -> "tilting"
            DetectedActivity.WALKING -> "walking"
            else -> "unknown"
        }
    }
}