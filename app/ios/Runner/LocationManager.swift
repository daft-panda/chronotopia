import CoreLocation
import Foundation
import UIKit

@objc public class LocationManager: NSObject, CLLocationManagerDelegate, FlutterPlugin {
    private var locationManager: CLLocationManager?
    private var eventSink: FlutterEventSink?
    public var locationUpdatesActive = false

    // Battery monitoring
    private var batteryLevel: Int = -1

    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(
            name: "io.chronotopia.app/location", binaryMessenger: registrar.messenger())
        let eventChannel = FlutterEventChannel(
            name: "io.chronotopia.app/location_stream", binaryMessenger: registrar.messenger()
        )

        let instance = LocationManager()
        registrar.addMethodCallDelegate(instance, channel: channel)
        eventChannel.setStreamHandler(instance)

        // Start UIDevice battery monitoring
        UIDevice.current.isBatteryMonitoringEnabled = true
        NotificationCenter.default.addObserver(
            instance,
            selector: #selector(batteryLevelDidChange),
            name: UIDevice.batteryLevelDidChangeNotification,
            object: nil
        )

        // Initial battery level reading
        instance.batteryLevel = Int(UIDevice.current.batteryLevel * 100)
    }

    @objc private func batteryLevelDidChange(_ notification: Notification) {
        batteryLevel = Int(UIDevice.current.batteryLevel * 100)
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "initialize":
            initializeLocationManager()
            result(nil)
        case "startTracking":
            startLocationUpdates()
            result(nil)
        case "stopTracking":
            stopLocationUpdates()
            result(nil)
        default:
            result(FlutterMethodNotImplemented)
        }
    }

    public func initializeLocationManager() {
        locationManager = CLLocationManager()
        locationManager?.delegate = self

        // Configure for background updates
        locationManager?.allowsBackgroundLocationUpdates = true
        locationManager?.pausesLocationUpdatesAutomatically = false
        locationManager?.showsBackgroundLocationIndicator = true

        // Set accuracy and update settings
        locationManager?.desiredAccuracy = kCLLocationAccuracyBest
        locationManager?.distanceFilter = 10  // 10 meters

        // Request permissions
        locationManager?.requestAlwaysAuthorization()

        // For visits monitoring
        locationManager?.startMonitoringVisits()

        // For significant location changes (power efficient)
        locationManager?.startMonitoringSignificantLocationChanges()
    }

    public func startLocationUpdates() {
        guard let locationManager = locationManager, !locationUpdatesActive else { return }

        // Start continuous updates when app is active
        locationManager.startUpdatingLocation()

        locationUpdatesActive = true

        // Also enable visit monitoring (helps with places detection)
        locationManager.startMonitoringVisits()

        // Register for system location updates
        if #available(iOS 14.0, *) {
            CLLocationManager.significantLocationChangeMonitoringAvailable()
        }
    }

    public func stopLocationUpdates() {
        guard let locationManager = locationManager, locationUpdatesActive else { return }

        locationManager.stopUpdatingLocation()
        locationManager.stopMonitoringVisits()
        locationManager.stopMonitoringSignificantLocationChanges()

        locationUpdatesActive = false
    }

    // MARK: - CLLocationManagerDelegate

    public func locationManager(
        _ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]
    ) {
        guard let location = locations.last else { return }

        // Create location data dictionary
        var locationData: [String: Any] = [
            "latitude": location.coordinate.latitude,
            "longitude": location.coordinate.longitude,
            "timestamp": ISO8601DateFormatter().string(from: location.timestamp),
            "batteryLevel": batteryLevel,
        ]

        // Add optional values with proper type checking
        if location.horizontalAccuracy >= 0 {
            locationData["accuracy"] = location.horizontalAccuracy
        }
        if location.verticalAccuracy >= 0 {
            locationData["verticalAccuracy"] = location.verticalAccuracy
        }
        if location.course >= 0 {
            locationData["heading"] = location.course
        }
        if location.speed >= 0 {
            locationData["speed"] = location.speed
        }
        if location.altitude != 0 {
            locationData["altitude"] = location.altitude
        }
        if #available(iOS 13.4, *), location.courseAccuracy >= 0 {
            locationData["headingAccuracy"] = location.courseAccuracy
        }
        if #available(iOS 10.0, *), location.speedAccuracy >= 0 {
            locationData["speedAccuracy"] = location.speedAccuracy
        }

        // Network type
        let networkInfo = getNetworkInfo()
        locationData["networkType"] = networkInfo

        // Floor level if available (for indoor locations)
        if let floor = location.floor {
            locationData["floor"] = floor.level
        }

        // Cache and send via gRPC
        GrpcClient.shared.cacheLocationData(locationData)

        // Also send to Flutter via event channel if available
        sendLocationData(locationData)
    }

    public func locationManager(_ manager: CLLocationManager, didVisit visit: CLVisit) {
        // Create visit data to track arrivals/departures
        var visitData: [String: Any] = [
            "latitude": visit.coordinate.latitude,
            "longitude": visit.coordinate.longitude,
            "horizontalAccuracy": visit.horizontalAccuracy,
        ]

        if visit.arrivalDate != Date.distantPast {
            visitData["arrivalTime"] = ISO8601DateFormatter().string(from: visit.arrivalDate)
        }

        if visit.departureDate != Date.distantFuture {
            visitData["departureTime"] = ISO8601DateFormatter().string(from: visit.departureDate)
        }

        // Send visit data to gRPC client
        GrpcClient.shared.cacheActivityData(visitData)

        // Also send to Flutter for UI updates
        sendVisitData(visitData)
    }

    public func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        NSLog("Location manager failed with error: \(error.localizedDescription)")
    }

    public func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        let status: CLAuthorizationStatus

        if #available(iOS 14.0, *) {
            status = manager.authorizationStatus
        } else {
            status = CLLocationManager.authorizationStatus()
        }

        switch status {
        case .authorizedAlways:
            // Ideal for background tracking
            startLocationUpdates()
        case .authorizedWhenInUse:
            // Request always authorization again with explanation
            locationManager?.requestAlwaysAuthorization()
        default:
            // Handle denied/restricted status
            break
        }
    }

    // MARK: - Helper Methods

    private func getNetworkInfo() -> String {
        let networkType = "Unknown"  // Default value

        // You'd need to use NWPathMonitor from Network framework for more details
        // This is simplified for this example

        return networkType
    }

    private func sendLocationData(_ data: [String: Any]) {
        // Convert to JSON string and send to Flutter
        if let jsonData = try? JSONSerialization.data(withJSONObject: data),
            let jsonString = String(data: jsonData, encoding: .utf8)
        {
            eventSink?(jsonString)
        }
    }

    private func sendVisitData(_ data: [String: Any]) {
        // Add a type field to distinguish it in the stream
        var visitData = data
        visitData["type"] = "visit"

        if let jsonData = try? JSONSerialization.data(withJSONObject: visitData),
            let jsonString = String(data: jsonData, encoding: .utf8)
        {
            eventSink?(jsonString)
        }
    }

    public func requestSingleLocationUpdate() {
        locationManager?.requestLocation()
    }

    public func setDesiredAccuracy(_ accuracy: CLLocationAccuracy) {
        locationManager?.desiredAccuracy = accuracy
    }

    public func setDistanceFilter(_ distance: CLLocationDistance) {
        locationManager?.distanceFilter = distance
    }
}

// MARK: - FlutterStreamHandler

extension LocationManager: FlutterStreamHandler {
    public func onListen(
        withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink
    ) -> FlutterError? {
        eventSink = events
        return nil
    }

    public func onCancel(withArguments arguments: Any?) -> FlutterError? {
        eventSink = nil
        return nil
    }
}
