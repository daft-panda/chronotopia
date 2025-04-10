import CoreMotion
// MotionActivity.swift
import Foundation
import UIKit

@objc public class MotionActivity: NSObject, FlutterPlugin {
    private let motionActivityManager = CMMotionActivityManager()
    private let pedometer = CMPedometer()
    private var eventSink: FlutterEventSink?
    private var isTracking = false

    // Activity tracking data
    private var lastStepCount: Int = 0
    private var lastStepDate: Date?

    // Current activity
    private var currentActivity: String = "unknown"
    private var currentConfidence: Int = 0
    private var activityStartTime: Date?

    // Background task
    private var backgroundTask: UIBackgroundTaskIdentifier = .invalid

    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(
            name: "io.chronotopia.app/activity", binaryMessenger: registrar.messenger())
        let eventChannel = FlutterEventChannel(
            name: "io.chronotopia.app/activity_stream", binaryMessenger: registrar.messenger())

        let instance = MotionActivity()
        registrar.addMethodCallDelegate(instance, channel: channel)
        eventChannel.setStreamHandler(instance)
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "initialize":
            checkAvailability(result)
        case "startTracking":
            startActivityTracking()
            result(nil)
        case "stopTracking":
            stopActivityTracking()
            result(nil)
        default:
            result(FlutterMethodNotImplemented)
        }
    }

    private func checkAvailability(_ result: @escaping FlutterResult) {
        let activityAvailable = CMMotionActivityManager.isActivityAvailable()
        let pedometerAvailable = CMPedometer.isStepCountingAvailable()

        let availabilityData: [String: Bool] = [
            "activityAvailable": activityAvailable,
            "pedometerAvailable": pedometerAvailable,
        ]

        result(availabilityData)
    }

    public func startActivityTracking() {
        guard !isTracking else { return }

        // Begin background task to get more execution time
        startBackgroundTask()

        // Start motion activity updates
        if CMMotionActivityManager.isActivityAvailable() {
            startActivityUpdates()
        }

        // Start pedometer for step counting
        if CMPedometer.isStepCountingAvailable() {
            startPedometerUpdates()
        }

        isTracking = true
    }

    public func stopActivityTracking() {
        guard isTracking else { return }

        // Stop motion activity updates
        motionActivityManager.stopActivityUpdates()

        // Stop pedometer updates
        pedometer.stopUpdates()
        pedometer.stopEventUpdates()

        isTracking = false

        // End background task
        endBackgroundTask()
    }

    private func startBackgroundTask() {
        backgroundTask = UIApplication.shared.beginBackgroundTask { [weak self] in
            self?.endBackgroundTask()
        }
    }

    private func endBackgroundTask() {
        if backgroundTask != .invalid {
            UIApplication.shared.endBackgroundTask(backgroundTask)
            backgroundTask = .invalid
        }
    }

    // MARK: - Activity Tracking

    public func startActivityUpdates() {
        let queue = OperationQueue()
        queue.qualityOfService = .utility

        motionActivityManager.startActivityUpdates(to: queue) { [weak self] (activity) in
            guard let self = self, let activity = activity else { return }

            // Process the activity
            self.handleMotionActivity(activity)
        }
    }

    private func handleMotionActivity(_ activity: CMMotionActivity) {
        // Get activity type
        let activityType = getActivityType(from: activity)

        // Get confidence level (0-100)
        let confidence = getConfidenceLevel(from: activity.confidence)

        // Check if this is a significant change
        if isSignificantActivityChange(newType: activityType, newConfidence: confidence) {
            // If we had a previous activity, send it with end time
            if let startTime = activityStartTime, currentActivity != "unknown" {
                let endTime = Date()
                sendActivityData(
                    type: currentActivity,
                    confidence: currentConfidence,
                    startTime: startTime,
                    endTime: endTime
                )
            }

            // Update current activity
            currentActivity = activityType
            currentConfidence = confidence
            activityStartTime = Date()
        }

        // Collect additional movement data
        collectMovementData(for: activityType)
    }

    private func getActivityType(from activity: CMMotionActivity) -> String {
        if activity.stationary {
            return "still"
        } else if activity.walking {
            return "walking"
        } else if activity.running {
            return "running"
        } else if activity.automotive {
            return "in_vehicle"
        } else if activity.cycling {
            return "on_bicycle"
        } else {
            return "unknown"
        }
    }

    private func getConfidenceLevel(from confidence: CMMotionActivityConfidence) -> Int {
        switch confidence {
        case .low:
            return 25
        case .medium:
            return 50
        case .high:
            return 100
        default:
            return 0
        }
    }

    private func isSignificantActivityChange(newType: String, newConfidence: Int) -> Bool {
        // Change in activity type is significant
        if newType != currentActivity {
            return true
        }

        // Significant change in confidence
        if abs(newConfidence - currentConfidence) > 20 {
            return true
        }

        // If it's been more than 5 minutes since we started this activity
        if let startTime = activityStartTime, Date().timeIntervalSince(startTime) > 300 {
            return true
        }

        return false
    }

    // MARK: - Pedometer Tracking

    private func startPedometerUpdates() {
        // Get live pedometer updates
        pedometer.startUpdates(from: Date()) { [weak self] (data, error) in
            guard let self = self, let data = data else { return }

            // Update step count
            self.lastStepCount = data.numberOfSteps.intValue
            self.lastStepDate = Date()

            // If we have a current activity, update its movement data
            if let startTime = self.activityStartTime {
                self.sendActivityData(
                    type: self.currentActivity,
                    confidence: self.currentConfidence,
                    startTime: startTime,
                    endTime: Date(),
                    steps: data.numberOfSteps.intValue,
                    distance: data.distance?.doubleValue
                )
            }
        }

        // Also monitor for pace changes
        if CMPedometer.isPaceAvailable() {
            pedometer.startEventUpdates { [weak self] (event, error) in
                guard let self = self, let event = event else { return }

                // Handle pace events (walking/running)
                if event.type == .pause {
                    self.sendPaceEvent(type: "pause")
                } else if event.type == .resume {
                    self.sendPaceEvent(type: "resume")
                }
            }
        }
    }

    private func sendPaceEvent(type: String) {
        // Send pace event data
        let paceData: [String: Any] = [
            "eventType": "pace",
            "paceStatus": type,
            "timestamp": ISO8601DateFormatter().string(from: Date()),
        ]

        // Send via event sink
        if let jsonData = try? JSONSerialization.data(withJSONObject: paceData),
            let jsonString = String(data: jsonData, encoding: .utf8)
        {
            DispatchQueue.main.async {
                self.eventSink?(jsonString)
            }
        }
    }

    // MARK: - Data Collection and Transmission

    private func collectMovementData(for activityType: String) {
        // For activities that don't have good automated tracking,
        // we can estimate movement data based on activity type and duration

        guard let startTime = activityStartTime else { return }

        let duration = Date().timeIntervalSince(startTime) / 60  // in minutes
        var steps: Int? = nil
        var distance: Double? = nil

        switch activityType {
        case "walking":
            // ~100 steps per minute, ~0.8m per step
            steps = Int(100 * duration)
            distance = 0.8 * Double(steps ?? 0)
        case "running":
            // ~150 steps per minute, ~1.0m per step
            steps = Int(150 * duration)
            distance = 1.0 * Double(steps ?? 0)
        case "on_bicycle":
            // Minimal steps, ~250m per minute
            steps = Int(10 * duration)
            distance = 250.0 * duration
        case "in_vehicle":
            // Minimal steps, variable distance (not reliable)
            steps = Int(5 * duration)
            distance = nil  // Not reliable
        default:
            // Minimal values for other activities
            steps = Int(10 * duration)
            distance = 10.0 * duration
        }

        // Send estimated data
        sendActivityData(
            type: activityType,
            confidence: currentConfidence,
            startTime: startTime,
            endTime: Date(),
            steps: steps,
            distance: distance
        )
    }

    private func sendActivityData(
        type: String,
        confidence: Int,
        startTime: Date,
        endTime: Date,
        steps: Int? = nil,
        distance: Double? = nil
    ) {
        // Create the activity data dictionary
        var activityData: [String: Any] = [
            "type": type,
            "confidence": confidence,
            "timestamp": ISO8601DateFormatter().string(from: endTime),
            "startTime": ISO8601DateFormatter().string(from: startTime),
            "endTime": ISO8601DateFormatter().string(from: endTime),
        ]

        // Add movement data if available
        if let steps = steps {
            activityData["stepCount"] = steps
        }

        if let distance = distance {
            activityData["distance"] = distance
        }

        // Cache and send via gRPC
        GrpcClient.shared.cacheActivityData(activityData)

        // Also send to Flutter via event channel if available
        if let jsonData = try? JSONSerialization.data(withJSONObject: activityData),
            let jsonString = String(data: jsonData, encoding: .utf8)
        {
            DispatchQueue.main.async {
                self.eventSink?(jsonString)
            }
        }
    }
}

// MARK: - FlutterStreamHandler

extension MotionActivity: FlutterStreamHandler {
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
