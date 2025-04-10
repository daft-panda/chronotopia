import BackgroundTasks
import CoreLocation
import Flutter
// iOS Method Channel Implementation
import Foundation
import UIKit
import UserNotifications

class MethodChannelHandler: NSObject {
    private var locationManager: LocationManager?
    private var motionActivity: MotionActivity?

    init(locationManager: LocationManager? = nil, motionActivity: MotionActivity? = nil) {
        self.locationManager = locationManager
        self.motionActivity = motionActivity
    }

    // Service Methods

    public func handleServiceMethods(call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "startServices":
            startServices()
            result(true)
        case "stopServices":
            stopServices()
            result(true)
        case "restartServices":
            restartServices()
            result(true)
        case "isServiceRunning":
            let isRunning = isServiceRunning()
            result(isRunning)
        case "setLowPowerMode":
            if let args = call.arguments as? [String: Any],
                let enabled = args["enabled"] as? Bool
            {
                setLowPowerMode(enabled: enabled)
                result(true)
            } else {
                result(
                    FlutterError(
                        code: "INVALID_ARGS", message: "Missing or invalid arguments", details: nil)
                )
            }
        case "applySettings":
            if let args = call.arguments as? [String: Any],
                let trackInLowBattery = args["trackInLowBattery"] as? Bool,
                let locationAccuracy = args["locationAccuracy"] as? Int,
                let uploadFrequency = args["uploadFrequency"] as? Int,
                let syncOnWifiOnly = args["syncOnWifiOnly"] as? Bool,
                let serverUrl = args["serverUrl"] as? String
            {

                applySettings(
                    trackInLowBattery: trackInLowBattery,
                    locationAccuracy: locationAccuracy,
                    uploadFrequency: uploadFrequency,
                    syncOnWifiOnly: syncOnWifiOnly,
                    serverUrl: serverUrl
                )
                result(true)
            } else {
                result(
                    FlutterError(
                        code: "INVALID_ARGS", message: "Missing or invalid arguments", details: nil)
                )
            }
        case "requestSingleLocationUpdate":
            requestSingleLocationUpdate()
            result(true)
        default:
            result(FlutterMethodNotImplemented)
        }
    }

    // Battery Methods

    public func handleBatteryMethods(call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "getBatteryLevel":
            let batteryLevel = getBatteryLevel()
            result(batteryLevel)
        case "isCharging":
            let isCharging = isCharging()
            result(isCharging)
        case "isIgnoringBatteryOptimizations":
            // iOS doesn't have battery optimizations like Android
            // Always return true to maintain compatibility
            result(true)
        case "requestBatteryOptimizationExemption":
            // Nothing to do on iOS
            result(true)
        default:
            result(FlutterMethodNotImplemented)
        }
    }

    // Settings Methods

    public func handleSettingsMethods(call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "openBatteryOptimizationSettings":
            openSettings()
            result(true)
        case "openAppSettings":
            openSettings()
            result(true)
        case "openLocationSettings":
            openSettings()
            result(true)
        default:
            result(FlutterMethodNotImplemented)
        }
    }

    // Implementation Methods

    private func startServices() {
        locationManager?.startLocationUpdates()
        motionActivity?.startActivityTracking()

        // Schedule background tasks
        scheduleBackgroundTasks()

        // Set up local notifications for service restarts
        setupNotifications()
    }

    private func stopServices() {
        locationManager?.stopLocationUpdates()
        motionActivity?.stopActivityTracking()
    }

    private func restartServices() {
        stopServices()
        startServices()
    }

    private func isServiceRunning() -> Bool {
        // In iOS, we can check if the location manager is actively tracking
        return locationManager?.locationUpdatesActive ?? false
    }

    private func setLowPowerMode(enabled: Bool) {
        // Adjust location accuracy based on low power mode
        let accuracy: CLLocationAccuracy =
            enabled ? kCLLocationAccuracyHundredMeters : kCLLocationAccuracyBest
        locationManager?.setDesiredAccuracy(accuracy)

        // Adjust update interval
        let interval: TimeInterval = enabled ? 60.0 : 10.0  // 1 minute vs 10 seconds
        locationManager?.setDistanceFilter(interval)
    }

    private func applySettings(
        trackInLowBattery: Bool,
        locationAccuracy: Int,
        uploadFrequency: Int,
        syncOnWifiOnly: Bool,
        serverUrl: String
    ) {
        // Apply location accuracy based on setting
        var accuracy: CLLocationAccuracy
        switch locationAccuracy {
        case 0:  // Low
            accuracy = kCLLocationAccuracyKilometer
        case 1:  // Balanced
            accuracy = kCLLocationAccuracyHundredMeters
        case 2:  // High
            accuracy = kCLLocationAccuracyBest
        default:
            accuracy = kCLLocationAccuracyHundredMeters
        }
        locationManager?.setDesiredAccuracy(accuracy)

        // Apply distance filter based on setting
        var distance: CLLocationDistance
        switch locationAccuracy {
        case 0:  // Low
            distance = 500.0  // 500 meters
        case 1:  // Balanced
            distance = 100.0  // 100 meters
        case 2:  // High
            distance = 10.0  // 10 meters
        default:
            distance = 100.0
        }
        locationManager?.setDistanceFilter(distance)

        // Store settings in UserDefaults
        let defaults = UserDefaults.standard
        defaults.set(trackInLowBattery, forKey: "track_in_low_battery")
        defaults.set(locationAccuracy, forKey: "location_accuracy")
        defaults.set(uploadFrequency, forKey: "upload_frequency")
        defaults.set(syncOnWifiOnly, forKey: "sync_on_wifi_only")
        defaults.set(serverUrl, forKey: "server_url")
        defaults.synchronize()
    }

    private func requestSingleLocationUpdate() {
        locationManager?.requestSingleLocationUpdate()
    }

    private func getBatteryLevel() -> Int {
        UIDevice.current.isBatteryMonitoringEnabled = true
        let level = UIDevice.current.batteryLevel

        // batteryLevel is between 0.0 and 1.0, or -1.0 if unknown
        if level < 0.0 {
            return -1
        } else {
            return Int(level * 100.0)
        }
    }

    private func isCharging() -> Bool {
        UIDevice.current.isBatteryMonitoringEnabled = true
        return UIDevice.current.batteryState == .charging || UIDevice.current.batteryState == .full
    }

    private func openSettings() {
        if let url = URL(string: UIApplication.openSettingsURLString) {
            if UIApplication.shared.canOpenURL(url) {
                UIApplication.shared.open(url, options: [:], completionHandler: nil)
            }
        }
    }

    // Background Task Management

    private func scheduleBackgroundTasks() {
        guard #available(iOS 13.0, *) else { return }

        // Register background processing task for periodic data sync
        let dataSyncTaskIdentifier = "com.example.trip_tracker.data_sync"
        BGTaskScheduler.shared.register(forTaskWithIdentifier: dataSyncTaskIdentifier, using: nil) {
            task in
            self.handleDataSyncTask(task as! BGProcessingTask)
        }

        // Register background refresh task for location updates
        let locationUpdateTaskIdentifier = "com.example.trip_tracker.location_update"
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: locationUpdateTaskIdentifier, using: nil
        ) { task in
            self.handleLocationUpdateTask(task as! BGAppRefreshTask)
        }

        // Schedule the tasks
        scheduleDataSyncTask()
        scheduleLocationUpdateTask()
    }

    @available(iOS 13.0, *)
    private func scheduleDataSyncTask() {
        let request = BGProcessingTaskRequest(identifier: "com.example.trip_tracker.data_sync")
        request.requiresNetworkConnectivity = true
        request.requiresExternalPower = false

        // Get the sync frequency from UserDefaults
        let defaults = UserDefaults.standard
        let uploadFrequency = defaults.integer(forKey: "upload_frequency")

        // Convert to actual time interval
        var earliestBeginDate: TimeInterval
        switch uploadFrequency {
        case 0:  // Low
            earliestBeginDate = 7200  // 2 hours
        case 1:  // Balanced
            earliestBeginDate = 1800  // 30 minutes
        case 2:  // High
            earliestBeginDate = 900  // 15 minutes
        default:
            earliestBeginDate = 1800
        }

        request.earliestBeginDate = Date(timeIntervalSinceNow: earliestBeginDate)

        do {
            try BGTaskScheduler.shared.submit(request)
        } catch {
            print("Could not schedule data sync task: \(error)")
        }
    }

    @available(iOS 13.0, *)
    private func scheduleLocationUpdateTask() {
        let request = BGAppRefreshTaskRequest(
            identifier: "com.example.trip_tracker.location_update")

        // Schedule for 15 minutes from now
        request.earliestBeginDate = Date(timeIntervalSinceNow: 900)

        do {
            try BGTaskScheduler.shared.submit(request)
        } catch {
            print("Could not schedule location update task: \(error)")
        }
    }

    @available(iOS 13.0, *)
    private func handleDataSyncTask(_ task: BGProcessingTask) {
        // Schedule the next sync task first
        scheduleDataSyncTask()

        // Create a task expiration handler
        task.expirationHandler = {
            // Clean up any pending work if the task is about to expire
        }

        // Perform the data sync
        // This would typically call your data sync service
        DispatchQueue.global(qos: .utility).async {
            // Simulate data sync
            Thread.sleep(forTimeInterval: 5)  // Simulate some work

            // Mark the task as complete
            task.setTaskCompleted(success: true)
        }
    }

    @available(iOS 13.0, *)
    private func handleLocationUpdateTask(_ task: BGAppRefreshTask) {
        // Schedule the next location update task first
        scheduleLocationUpdateTask()

        // Create a task expiration handler
        task.expirationHandler = {
            // Clean up any pending work if the task is about to expire
        }

        // Request a single location update
        self.requestSingleLocationUpdate()

        // Wait for location update completion or timeout
        DispatchQueue.global(qos: .utility).async {
            // Give it a short time to get location
            Thread.sleep(forTimeInterval: 10)

            // Mark the task as complete
            task.setTaskCompleted(success: true)
        }
    }

    // Notification Setup

    private func setupNotifications() {
        let center = UNUserNotificationCenter.current()
        center.requestAuthorization(options: [.alert, .sound]) { granted, error in
            if let error = error {
                print("Notification authorization error: \(error)")
            }
        }

        // Create a "tracking restarted" notification
        let content = UNMutableNotificationContent()
        content.title = "Tracking Restarted"
        content.body = "Trip Tracker has restarted location tracking"
        content.sound = UNNotificationSound.default

        // Create a time-based trigger that repeats daily
        // This acts as a backup to ensure tracking is active
        let triggerDaily = UNCalendarNotificationTrigger(
            dateMatching: Calendar.current.dateComponents(
                [.hour, .minute], from: Date(timeIntervalSinceNow: 86400)),
            repeats: true
        )

        let request = UNNotificationRequest(
            identifier: "com.example.trip_tracker.tracking_restart",
            content: content,
            trigger: triggerDaily
        )

        center.add(request) { error in
            if let error = error {
                print("Error scheduling notification: \(error)")
            }
        }
    }
}

// Extension for BGTasks on iOS 13+
@available(iOS 13.0, *)
extension MethodChannelHandler {
    private func handleBackgroundTask(_ task: BGTask) {
        // Check if tracking is enabled
        let defaults = UserDefaults.standard
        let isTrackingEnabled = defaults.bool(forKey: "tracking_enabled")

        if !isTrackingEnabled {
            task.setTaskCompleted(success: true)
            return
        }

        if task is BGProcessingTask {
            handleDataSyncTask(task as! BGProcessingTask)
        } else if task is BGAppRefreshTask {
            handleLocationUpdateTask(task as! BGAppRefreshTask)
        }
    }
}
