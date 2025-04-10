// GrpcClient.kt
package com.example.trip_tracker

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Build
import android.util.Log
import io.grpc.ManagedChannel
import io.grpc.ManagedChannelBuilder
import io.grpc.stub.StreamObserver
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.FileReader
import java.net.InetSocketAddress
import java.net.Socket
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import com.example.trip_tracker.grpc.*
import java.lang.Exception
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.atomic.AtomicBoolean

class GrpcClient(private val context: Context) {
    private val TAG = "GrpcClient"
    
    // Server connection settings
    private val serverHost = "your-server-host.com"
    private val serverPort = 443
    private val useSSL = true
    
    // gRPC channel and stubs
    private var channel: ManagedChannel? = null
    private var stub: TripTrackerServiceGrpc.TripTrackerServiceStub? = null
    private var blockingStub: TripTrackerServiceGrpc.TripTrackerServiceBlockingStub? = null
    
    // Stream observer for bidirectional streaming
    private var responseObserver: StreamObserver<DataResponse>? = null
    private var requestObserver: StreamObserver<DataPacket>? = null
    
    // Thread pool for background work
    private val executor = Executors.newSingleThreadExecutor()
    
    // Cache directories
    private val locationCacheDir: File
    private val activityCacheDir: File
    
    // Sync state
    private val isSyncing = AtomicBoolean(false)
    private var lastSyncAttempt = 0L
    
    // User and device info
    private var userId: String? = null
    private var deviceId: String? = null
    private var deviceMetadata: DeviceMetadata? = null
    
    init {
        // Create cache directories
        locationCacheDir = File(context.filesDir, "location_cache")
        if (!locationCacheDir.exists()) {
            locationCacheDir.mkdirs()
        }
        
        activityCacheDir = File(context.filesDir, "activity_cache")
        if (!activityCacheDir.exists()) {
            activityCacheDir.mkdirs()
        }
        
        // Load or create user ID and device ID
        loadUserAndDeviceInfo()
        
        // Initialize device metadata
        initDeviceMetadata()
    }
    
    private fun loadUserAndDeviceInfo() {
        val prefs = context.getSharedPreferences("com.example.trip_tracker", Context.MODE_PRIVATE)
        
        userId = prefs.getString("user_id", null)
        if (userId == null) {
            userId = "user_${System.currentTimeMillis()}"
            prefs.edit().putString("user_id", userId).apply()
        }
        
        deviceId = prefs.getString("device_id", null)
        if (deviceId == null) {
            deviceId = "device_${System.currentTimeMillis()}"
            prefs.edit().putString("device_id", deviceId).apply()
        }
    }
    
    private fun initDeviceMetadata() {
        try {
            deviceMetadata = DeviceMetadata.newBuilder()
                .setPlatform("android")
                .setOsVersion(Build.VERSION.RELEASE)
                .setDeviceModel("${Build.MANUFACTURER} ${Build.MODEL}")
                .setTimezone(TimeZone.getDefault().id)
                .setLocale(Locale.getDefault().toString())
                .setDeviceLanguage(Locale.getDefault().language)
                .build()
            
            // Try to get app version
            try {
                val packageInfo = context.packageManager.getPackageInfo(context.packageName, 0)
                val appVersion = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    packageInfo.longVersionCode.toString()
                } else {
                    @Suppress("DEPRECATION")
                    packageInfo.versionCode.toString()
                }
                
                deviceMetadata = deviceMetadata?.toBuilder()
                    ?.setAppVersion(appVersion)
                    ?.build()
            } catch (e: Exception) {
                Log.e(TAG, "Error getting app version", e)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing device metadata", e)
        }
    }
    
    fun cacheLocationData(locationData: JSONObject) {
        try {
            // Generate filename with timestamp
            val timestamp = System.currentTimeMillis()
            val filename = "location_$timestamp.json"
            val file = File(locationCacheDir, filename)
            
            // Write data to file
            file.writeText(locationData.toString())
            
            // Try to sync if we have a connection
            if (hasNetworkConnection()) {
                scheduleSyncData()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error caching location data", e)
        }
    }
    
    fun cacheActivityData(activityData: JSONObject) {
        try {
            // Generate filename with timestamp
            val timestamp = System.currentTimeMillis()
            val filename = "activity_$timestamp.json"
            val file = File(activityCacheDir, filename)
            
            // Write data to file
            file.writeText(activityData.toString())
            
            // Try to sync if we have a connection
            if (hasNetworkConnection()) {
                scheduleSyncData()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error caching activity data", e)
        }
    }
    
    fun scheduleSyncData() {
        // Don't schedule if we're already syncing
        if (isSyncing.get()) {
            return
        }
        
        // Don't sync too frequently - at most once per minute
        val now = System.currentTimeMillis()
        if (now - lastSyncAttempt < 60000) {
            return
        }
        
        lastSyncAttempt = now
        
        // Submit sync task to executor
        executor.submit {
            syncData()
        }
    }
    
    private fun syncData() {
        if (isSyncing.getAndSet(true)) {
            return
        }
        
        try {
            // Check for network connection
            if (!hasNetworkConnection()) {
                Log.d(TAG, "No network connection, skipping sync")
                isSyncing.set(false)
                return
            }
            
            // Check server reachability
            if (!isServerReachable()) {
                Log.d(TAG, "Server not reachable, skipping sync")
                isSyncing.set(false)
                return
            }
            
            // Initialize gRPC channel if needed
            if (channel == null || channel?.isShutdown == true) {
                initializeGrpc()
            }
            
            // Process cached location data
            val locationFiles = locationCacheDir.listFiles()
            val activityFiles = activityCacheDir.listFiles()
            
            if ((locationFiles == null || locationFiles.isEmpty()) && 
                (activityFiles == null || activityFiles.isEmpty())) {
                Log.d(TAG, "No cached data to sync")
                isSyncing.set(false)
                return
            }
            
            // Start bidirectional stream if not already started
            if (requestObserver == null) {
                initDataStream()
            }
            
            // Batch data for efficient sending
            val batchSize = 50
            val locationBatches = locationFiles?.toList()?.chunked(batchSize) ?: listOf()
            val activityBatches = activityFiles?.toList()?.chunked(batchSize) ?: listOf()
            
            // Process location batches
            for (batch in locationBatches) {
                val dataPacket = buildDataPacket()
                
                for (file in batch) {
                    try {
                        val jsonString = file.readText()
                        val jsonObject = JSONObject(jsonString)
                        
                        val locationPoint = parseLocationPoint(jsonObject)
                        dataPacket.addLocations(locationPoint)
                        
                        // Mark file for deletion if successfully added to packet
                        file.delete()
                    } catch (e: Exception) {
                        Log.e(TAG, "Error processing location file: ${file.name}", e)
                        // Move to error directory if persistent error
                        if (file.lastModified() < System.currentTimeMillis() - 86400000) { // Older than 1 day
                            val errorDir = File(context.filesDir, "location_errors")
                            if (!errorDir.exists()) {
                                errorDir.mkdirs()
                            }
                            file.renameTo(File(errorDir, file.name))
                        }
                    }
                }
                
                // Send the batch if it has any locations
                if (dataPacket.locationsCount > 0) {
                    sendDataPacket(dataPacket)
                }
            }
            
            // Process activity batches
            for (batch in activityBatches) {
                val dataPacket = buildDataPacket()
                
                for (file in batch) {
                    try {
                        val jsonString = file.readText()
                        val jsonObject = JSONObject(jsonString)
                        
                        val activityEvent = parseActivityEvent(jsonObject)
                        dataPacket.addActivities(activityEvent)
                        
                        // Mark file for deletion if successfully added to packet
                        file.delete()
                    } catch (e: Exception) {
                        Log.e(TAG, "Error processing activity file: ${file.name}", e)
                        // Move to error directory if persistent error
                        if (file.lastModified() < System.currentTimeMillis() - 86400000) { // Older than 1 day
                            val errorDir = File(context.filesDir, "activity_errors")
                            if (!errorDir.exists()) {
                                errorDir.mkdirs()
                            }
                            file.renameTo(File(errorDir, file.name))
                        }
                    }
                }
                
                // Send the batch if it has any activities
                if (dataPacket.activitiesCount > 0) {
                    sendDataPacket(dataPacket)
                }
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error during data sync", e)
        } finally {
            isSyncing.set(false)
        }
    }
    
    private fun buildDataPacket(): DataPacket.Builder {
        return DataPacket.newBuilder()
            .setUserId(userId ?: "unknown_user")
            .setDeviceId(deviceId ?: "unknown_device")
            .setTimestamp(System.currentTimeMillis())
            .setDeviceMetadata(deviceMetadata ?: DeviceMetadata.getDefaultInstance())
    }
    
    private fun parseLocationPoint(json: JSONObject): LocationPoint {
        val builder = LocationPoint.newBuilder()
        
        builder.latitude = json.getDouble("latitude")
        builder.longitude = json.getDouble("longitude")
        
        if (json.has("altitude") && !json.isNull("altitude")) {
            builder.altitude = json.getDouble("altitude")
        }
        
        if (json.has("accuracy") && !json.isNull("accuracy")) {
            builder.horizontalAccuracy = json.getDouble("accuracy")
        }
        
        if (json.has("verticalAccuracy") && !json.isNull("verticalAccuracy")) {
            builder.verticalAccuracy = json.getDouble("verticalAccuracy")
        }
        
        if (json.has("speed") && !json.isNull("speed")) {
            builder.speed = json.getDouble("speed")
        }
        
        if (json.has("speedAccuracy") && !json.isNull("speedAccuracy")) {
            builder.speedAccuracy = json.getDouble("speedAccuracy")
        }
        
        if (json.has("heading") && !json.isNull("heading")) {
            builder.bearing = json.getDouble("heading")
        }
        
        if (json.has("bearingAccuracy") && !json.isNull("bearingAccuracy")) {
            builder.bearingAccuracy = json.getDouble("bearingAccuracy")
        }
        
        if (json.has("timestamp")) {
            val timestampStr = json.getString("timestamp")
            try {
                val sdf = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US)
                sdf.timeZone = TimeZone.getTimeZone("UTC")
                val date = sdf.parse(timestampStr)
                builder.timestamp = date?.time ?: System.currentTimeMillis()
            } catch (e: Exception) {
                builder.timestamp = System.currentTimeMillis()
            }
        } else {
            builder.timestamp = System.currentTimeMillis()
        }
        
        if (json.has("provider") && !json.isNull("provider")) {
            builder.provider = json.getString("provider")
        }
        
        if (json.has("isMockLocation")) {
            builder.isMockLocation = json.getBoolean("isMockLocation")
        }
        
        if (json.has("floor") && !json.isNull("floor")) {
            builder.floorLevel = json.getInt("floor")
        }
        
        if (json.has("batteryLevel")) {
            builder.batteryLevel = json.getInt("batteryLevel")
        }
        
        if (json.has("networkType") && !json.isNull("networkType")) {
            builder.networkType = json.getString("networkType")
        }
        
        return builder.build()
    }
    
    private fun parseActivityEvent(json: JSONObject): ActivityEvent {
        val builder = ActivityEvent.newBuilder()
        
        // Parse activity type
        val activityTypeStr = json.getString("type")
        val activityType = when (activityTypeStr) {
            "in_vehicle" -> ActivityEvent.ActivityType.IN_VEHICLE
            "on_bicycle" -> ActivityEvent.ActivityType.ON_BICYCLE
            "on_foot" -> ActivityEvent.ActivityType.ON_FOOT
            "running" -> ActivityEvent.ActivityType.RUNNING
            "still" -> ActivityEvent.ActivityType.STILL
            "tilting" -> ActivityEvent.ActivityType.TILTING
            "walking" -> ActivityEvent.ActivityType.WALKING
            else -> ActivityEvent.ActivityType.UNKNOWN
        }
        builder.setType(activityType)
        
        // Parse confidence
        if (json.has("confidence")) {
            builder.confidence = json.getInt("confidence")
        }
        
        // Parse timestamps
        if (json.has("timestamp")) {
            builder.timestamp = json.getLong("timestamp")
        } else {
            builder.timestamp = System.currentTimeMillis()
        }
        
        if (json.has("startTime")) {
            builder.startTime = json.getLong("startTime")
        }
        
        if (json.has("endTime")) {
            builder.endTime = json.getLong("endTime")
        }
        
        // Parse movement data
        if (json.has("stepCount") && !json.isNull("stepCount")) {
            builder.stepCount = json.getLong("stepCount")
        }
        
        if (json.has("distance") && !json.isNull("distance")) {
            builder.distance = json.getDouble("distance")
        }
        
        return builder.build()
    }
    
    private fun hasNetworkConnection(): Boolean {
        val connectivityManager = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val network = connectivityManager.activeNetwork ?: return false
            val capabilities = connectivityManager.getNetworkCapabilities(network) ?: return false
            
            return capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
        } else {
            @Suppress("DEPRECATION")
            val networkInfo = connectivityManager.activeNetworkInfo
            return networkInfo != null && networkInfo.isConnected
        }
    }
    
    private fun isServerReachable(): Boolean {
        return try {
            val socket = Socket()
            socket.connect(InetSocketAddress(serverHost, serverPort), 5000)
            socket.close()
            true
        } catch (e: Exception) {
            Log.e(TAG, "Server not reachable", e)
            false
        }
    }
    
    private fun initializeGrpc() {
        try {
            // Create the channel
            val channelBuilder = ManagedChannelBuilder.forAddress(serverHost, serverPort)
            
            if (useSSL) {
                channelBuilder.useTransportSecurity()
            } else {
                channelBuilder.usePlaintext()
            }
            
            channel = channelBuilder
                .keepAliveTime(30, TimeUnit.SECONDS)
                .keepAliveTimeout(10, TimeUnit.SECONDS)
                .build()
            
            // Create stubs
            stub = TripTrackerServiceGrpc.newStub(channel)
            blockingStub = TripTrackerServiceGrpc.newBlockingStub(channel)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing gRPC", e)
            throw e
        }
    }
    
    private fun initDataStream() {
        try {
            responseObserver = object : StreamObserver<DataResponse> {
                override fun onNext(response: DataResponse) {
                    Log.d(TAG, "Server response: ${response.message}")
                    
                    // Handle server instructions
                    if (response.pauseTracking) {
                        // Implement pause tracking logic if needed
                    }
                    
                    if (response.recommendedUploadInterval > 0) {
                        // Adjust upload frequency based on server recommendation
                    }
                }
                
                override fun onError(t: Throwable) {
                    Log.e(TAG, "Error in gRPC stream", t)
                    requestObserver = null
                    // Try to reestablish connection later
                }
                
                override fun onCompleted() {
                    Log.d(TAG, "gRPC stream completed")
                    requestObserver = null
                }
            }
            
            requestObserver = stub?.sendData(responseObserver)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing data stream", e)
            requestObserver = null
            throw e
        }
    }
    
    private fun sendDataPacket(dataPacket: DataPacket.Builder) {
        try {
            val packet = dataPacket.build()
            
            if (requestObserver == null) {
                initDataStream()
            }
            
            requestObserver?.onNext(packet)
            Log.d(TAG, "Sent data packet with ${packet.locationsCount} locations and ${packet.activitiesCount} activities")
            
        } catch (e: Exception) {
            Log.e(TAG, "Error sending data packet", e)
            requestObserver = null
            
            // If streaming fails, retry with blocking stub
            try {
                val response = blockingStub?.sendData(listOf(dataPacket.build()).iterator())
                Log.d(TAG, "Sent data with blocking stub, response: ${response?.message}")
            } catch (e2: Exception) {
                Log.e(TAG, "Error sending data with blocking stub", e2)
                // If both fail, we'll retry on next sync attempt
            }
        }
    }
    
    fun shutdown() {
        try {
            // Complete the stream if it exists
            requestObserver?.onCompleted()
            
            // Shutdown the channel
            channel?.shutdown()?.awaitTermination(5, TimeUnit.SECONDS)
            
            // Shutdown executor
            executor.shutdown()
            executor.awaitTermination(5, TimeUnit.SECONDS)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error shutting down gRPC client", e)
        }
    }
}