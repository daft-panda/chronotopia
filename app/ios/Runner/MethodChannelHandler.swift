import CoreLocation
import Flutter
// iOS Method Channel Implementation
import Foundation
import UIKit
import UserNotifications

class MethodChannelHandler: NSObject {
    private var locationManager: LocationManager?
    private var motionActivity: MotionActivity?
    private var grpcClient: GrpcClient?

    init(locationManager: LocationManager? = nil, motionActivity: MotionActivity? = nil) {
        self.locationManager = locationManager
        self.motionActivity = motionActivity
        self.grpcClient = GrpcClient.shared
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
        case "getUploadStatus":
            let status = getUploadStatus()
            result(status)
        case "setAuthToken":
            if let args = call.arguments as? [String: Any],
                let token = args["token"] as? String
            {
                setAuthToken(token)
                result(true)
            } else {
                result(
                    FlutterError(
                        code: "INVALID_ARGS", message: "Missing or invalid token", details: nil)
                )
            }
        case "setServerSettings":
            if let args = call.arguments as? [String: Any],
                let serverUrl = args["serverUrl"] as? String,
                let useTLS = args["useTLS"] as? Bool
            {
                setServerSettings(serverUrl: serverUrl, useTLS: useTLS)
                result(true)
            } else {
                result(
                    FlutterError(
                        code: "INVALID_ARGS", message: "Missing or invalid server settings",
                        details: nil)
                )
            }
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
                // Get auth token and TLS setting if provided
                let authToken = args["authToken"] as? String
                let useTLS = args["useTLS"] as? Bool ?? true

                applySettings(
                    trackInLowBattery: trackInLowBattery,
                    locationAccuracy: locationAccuracy,
                    uploadFrequency: uploadFrequency,
                    syncOnWifiOnly: syncOnWifiOnly,
                    serverUrl: serverUrl,
                    useTLS: useTLS,
                    authToken: authToken
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
        case "forceSyncNow":
            forceSyncNow()
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

    private func getUploadStatus() -> [String: Any] {
        // Get status from gRPC client
        if let status = grpcClient?.getUploadStatus() {
            return status
        }
        return ["error": "Unable to get upload status"]
    }

    private func forceSyncNow() {
        // Force an immediate sync
        grpcClient?.scheduleSyncData()
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

    private func setAuthToken(_ token: String) {
        // Set the token in the gRPC client
        GrpcClient.shared.setAuthToken(token)

        // Store in user defaults for persistence
        let defaults = UserDefaults.standard
        defaults.set(token, forKey: "auth_token")
        defaults.synchronize()
    }

    private func setServerSettings(serverUrl: String, useTLS: Bool) {
        // Store in user defaults for persistence
        let defaults = UserDefaults.standard
        defaults.set(serverUrl, forKey: "server_url")
        defaults.set(useTLS, forKey: "use_tls")
        defaults.synchronize()

        // Update the server settings in the GrpcClient
        GrpcClient.shared.setServerSettings(serverUrl: serverUrl, useTLS: useTLS)
    }

    private func applySettings(
        trackInLowBattery: Bool,
        locationAccuracy: Int,
        uploadFrequency: Int,
        syncOnWifiOnly: Bool,
        serverUrl: String,
        useTLS: Bool = true,
        authToken: String? = nil
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
        defaults.set(useTLS, forKey: "use_tls")

        // Update server settings
        setServerSettings(serverUrl: serverUrl, useTLS: useTLS)

        // Update auth token if provided
        if let authToken = authToken {
            setAuthToken(authToken)
        }

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
        content.title = "Tracking Active"
        content.body = "Chronotopia is tracking your location"
        content.sound = UNNotificationSound.default

        let request = UNNotificationRequest(
            identifier: "io.chronotopia.app.tracking_active",
            content: content,
            trigger: nil  // No trigger means it shows immediately
        )

        center.add(request) { error in
            if let error = error {
                print("Error showing notification: \(error)")
            }
        }
    }
}
