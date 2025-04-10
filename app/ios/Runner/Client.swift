import CoreLocation
// GrpcClient.swift
import Foundation
import GRPCCore
import GRPCNIOTransportHTTP2
import NIO
import Network
import SwiftProtobuf
import UIKit

class GrpcClient {
    // Server connection settings
    private let serverHost: String
    private let serverPort = 443
    private let useSSL = true

    // gRPC client
    private var client: GRPCClient<HTTP2ClientTransport.Posix>?

    // Trip Tracker client
    private var tripTrackerClient: Chronotopia_Ingest.Client<HTTP2ClientTransport.Posix>?

    // User and device info
    private var userId: String?
    private var deviceId: String?
    private var deviceMetadata: Chronotopia_DeviceMetadata?

    // Cache directories
    private let locationCacheURL: URL
    private let activityCacheURL: URL

    // Sync state
    private var isSyncing = false
    private var lastSyncAttempt: Date?

    // Queue for serializing access
    private let queue = DispatchQueue(
        label: "io.chronotopia.app.grpc", qos: .utility)

    // Network monitor
    private let networkMonitor = NWPathMonitor()
    private var hasNetworkConnection = false

    // Task management
    private var syncTask: Task<Void, Error>?
    private var clientTask: Task<Void, Error>?

    // Singleton instance
    static let shared = GrpcClient()

    private init() {
        let defaults = UserDefaults.standard
        serverHost = defaults.string(forKey: "server_url") ?? "ingest.chronotopia.io"
        // Create cache directories
        let fileManager = FileManager.default
        let documentDirectory = fileManager.urls(
            for: .documentDirectory, in: .userDomainMask
        )
        .first!

        locationCacheURL = documentDirectory.appendingPathComponent(
            "location_cache")
        try? fileManager.createDirectory(
            at: locationCacheURL, withIntermediateDirectories: true)

        activityCacheURL = documentDirectory.appendingPathComponent(
            "activity_cache")
        try? fileManager.createDirectory(
            at: activityCacheURL, withIntermediateDirectories: true)

        // Load or create user and device IDs
        loadUserAndDeviceInfo()

        // Initialize device metadata
        initDeviceMetadata()

        // Set up network monitoring
        setupNetworkMonitoring()
    }

    private func loadUserAndDeviceInfo() {
        let defaults = UserDefaults.standard

        userId = defaults.string(forKey: "user_id")
        if userId == nil {
            userId = "user_\(Int(Date().timeIntervalSince1970 * 1000))"
            defaults.set(userId, forKey: "user_id")
        }

        deviceId = defaults.string(forKey: "device_id")
        if deviceId == nil {
            deviceId = "device_\(Int(Date().timeIntervalSince1970 * 1000))"
            defaults.set(deviceId, forKey: "device_id")
        }
    }

    private func initDeviceMetadata() {
        deviceMetadata = Chronotopia_DeviceMetadata()
        deviceMetadata?.platform = "ios"
        deviceMetadata?.osVersion = UIDevice.current.systemVersion
        deviceMetadata?.deviceModel = UIDevice.current.model
        deviceMetadata?.deviceLanguage = Locale.current.language.languageCode?.identifier ?? "en"

        // Try to get app version
        if let appVersion = Bundle.main.infoDictionary?[
            "CFBundleShortVersionString"] as? String
        {
            deviceMetadata?.appVersion = appVersion
        }
    }

    private func setupNetworkMonitoring() {
        networkMonitor.pathUpdateHandler = { [weak self] path in
            self?.hasNetworkConnection = path.status == .satisfied

            // If we get a network connection and have cached data, try to sync
            if path.status == .satisfied {
                self?.scheduleSyncData()
            }
        }

        let queue = DispatchQueue(label: "NetworkMonitor")
        networkMonitor.start(queue: queue)
    }

    func cacheLocationData(_ locationData: [String: Any]) {
        queue.async {
            do {
                // Generate filename with timestamp
                let timestamp = Int(Date().timeIntervalSince1970 * 1000)
                let filename = "location_\(timestamp).json"
                let fileURL = self.locationCacheURL.appendingPathComponent(
                    filename)

                // Convert to JSON data
                let jsonData = try JSONSerialization.data(
                    withJSONObject: locationData)

                // Write to file
                try jsonData.write(to: fileURL)

                // Try to sync if we have a connection
                if self.hasNetworkConnection {
                    self.scheduleSyncData()
                }
            } catch {
                print("Error caching location data: \(error)")
            }
        }
    }

    func cacheActivityData(_ activityData: [String: Any]) {
        queue.async {
            do {
                // Generate filename with timestamp
                let timestamp = Int(Date().timeIntervalSince1970 * 1000)
                let filename = "activity_\(timestamp).json"
                let fileURL = self.activityCacheURL.appendingPathComponent(
                    filename)

                // Convert to JSON data
                let jsonData = try JSONSerialization.data(
                    withJSONObject: activityData)

                // Write to file
                try jsonData.write(to: fileURL)

                // Try to sync if we have a connection
                if self.hasNetworkConnection {
                    self.scheduleSyncData()
                }
            } catch {
                print("Error caching activity data: \(error)")
            }
        }
    }

    func scheduleSyncData() {
        Task {
            await scheduleSyncDataInternal()
        }
    }

    private func scheduleSyncDataInternal() async {
        await queue.async {
            // Don't schedule if we're already syncing
            if self.isSyncing {
                return
            }

            // Don't sync too frequently - at most once per minute
            if let lastSyncAttempt = self.lastSyncAttempt,
                Date().timeIntervalSince(lastSyncAttempt) < 60
            {
                return
            }

            self.lastSyncAttempt = Date()

            // Cancel any existing task
            self.syncTask?.cancel()

            // Create a new task for syncing
            self.syncTask = Task {
                do {
                    try await self.syncData()
                } catch {
                    print("Error syncing data: \(error)")
                }
            }
        }
    }

    private func syncData() async throws {
        // Set syncing flag
        await queue.async {
            if self.isSyncing {
                return
            }
            self.isSyncing = true
        }

        defer {
            Task {
                await queue.async {
                    self.isSyncing = false
                }
            }
        }

        // Check for network connection
        guard hasNetworkConnection else {
            print("No network connection, skipping sync")
            return
        }

        // Check if client is available
        guard let tripTrackerClient = tripTrackerClient else {
            print("gRPC client not initialized")
            return
        }

        // Check for cached files
        let fileManager = FileManager.default

        do {
            // Get location files
            let locationFiles = try fileManager.contentsOfDirectory(
                at: locationCacheURL, includingPropertiesForKeys: nil)

            // Get activity files
            let activityFiles = try fileManager.contentsOfDirectory(
                at: activityCacheURL, includingPropertiesForKeys: nil)

            if locationFiles.isEmpty && activityFiles.isEmpty {
                print("No cached data to sync")
                return
            }

            // Process in batches for efficiency
            let batchSize = 50
            let locationBatches = locationFiles.chunked(into: batchSize)
            let activityBatches = activityFiles.chunked(into: batchSize)

            // Process location batches
            for batch in locationBatches {
                // Create data packet
                var dataPacket = Chronotopia_IngestBatch()
                dataPacket.dateTime = now()

                if let metadata = deviceMetadata {
                    dataPacket.deviceMetadata = metadata
                }

                // Process each file in batch
                for fileURL in batch {
                    do {
                        let jsonData = try Data(contentsOf: fileURL)
                        let jsonObject =
                            try JSONSerialization.jsonObject(with: jsonData)
                            as! [String: Any]

                        let locationPoint = parseLocationPoint(jsonObject)
                        dataPacket.locations.append(locationPoint)

                        // Delete file after successful processing
                        try fileManager.removeItem(at: fileURL)
                    } catch {
                        print(
                            "Error processing location file \(fileURL.lastPathComponent): \(error)"
                        )

                        // Move to error directory if persistent error
                        if let attributes = try? fileManager.attributesOfItem(
                            atPath: fileURL.path),
                            let creationDate = attributes[.creationDate]
                                as? Date,
                            creationDate.timeIntervalSinceNow < -86400
                        {  // Older than 1 day

                            let errorDir =
                                locationCacheURL.appendingPathComponent(
                                    "errors")
                            try? fileManager.createDirectory(
                                at: errorDir, withIntermediateDirectories: true)
                            let errorFile = errorDir.appendingPathComponent(
                                fileURL.lastPathComponent)
                            try? fileManager.moveItem(
                                at: fileURL, to: errorFile)
                        }
                    }
                }

                // Send the batch if it has any locations
                if !dataPacket.locations.isEmpty {
                    try await sendDataPacket(
                        dataPacket, using: tripTrackerClient)
                }
            }

            // Process activity batches
            for batch in activityBatches {
                // Create data packet
                var dataPacket = Chronotopia_IngestBatch()
                dataPacket.dateTime = now()

                if let metadata = deviceMetadata {
                    dataPacket.deviceMetadata = metadata
                }

                // Process each file in batch
                for fileURL in batch {
                    do {
                        let jsonData = try Data(contentsOf: fileURL)
                        let jsonObject =
                            try JSONSerialization.jsonObject(with: jsonData)
                            as! [String: Any]

                        let activityEvent = parseActivityEvent(jsonObject)
                        dataPacket.activities.append(activityEvent)

                        // Delete file after successful processing
                        try fileManager.removeItem(at: fileURL)
                    } catch {
                        print(
                            "Error processing activity file \(fileURL.lastPathComponent): \(error)"
                        )

                        // Move to error directory if persistent error
                        if let attributes = try? fileManager.attributesOfItem(
                            atPath: fileURL.path),
                            let creationDate = attributes[.creationDate]
                                as? Date,
                            creationDate.timeIntervalSinceNow < -86400
                        {  // Older than 1 day

                            let errorDir =
                                activityCacheURL.appendingPathComponent(
                                    "errors")
                            try? fileManager.createDirectory(
                                at: errorDir, withIntermediateDirectories: true)
                            let errorFile = errorDir.appendingPathComponent(
                                fileURL.lastPathComponent)
                            try? fileManager.moveItem(
                                at: fileURL, to: errorFile)
                        }
                    }
                }

                // Send the batch if it has any activities
                if !dataPacket.activities.isEmpty {
                    try await sendDataPacket(
                        dataPacket, using: tripTrackerClient)
                }
            }

        } catch {
            print("Error during data sync: \(error)")
            throw error
        }
    }

    private func parseLocationPoint(_ json: [String: Any])
        -> Chronotopia_LocationPoint
    {
        var locationPoint = Chronotopia_LocationPoint()

        locationPoint.latitude = json["latitude"] as? Double ?? 0
        locationPoint.longitude = json["longitude"] as? Double ?? 0

        if let altitude = json["altitude"] as? Double {
            locationPoint.altitude = altitude
        }

        if let accuracy = json["accuracy"] as? Double {
            locationPoint.horizontalAccuracy = accuracy
        }

        if let verticalAccuracy = json["verticalAccuracy"] as? Double {
            locationPoint.verticalAccuracy = verticalAccuracy
        }

        if let heading = json["heading"] as? Double {
            locationPoint.bearing = heading
        }

        if let bearingAccuracy = json["headingAccuracy"] as? Double {
            locationPoint.bearingAccuracy = bearingAccuracy
        }

        if let speed = json["speed"] as? Double {
            locationPoint.speed = speed
        }

        if let speedAccuracy = json["speedAccuracy"] as? Double {
            locationPoint.speedAccuracy = speedAccuracy
        }

        if let timestampStr = json["timestamp"] as? String {
            let formatter = ISO8601DateFormatter()
            if let date = formatter.date(from: timestampStr) {
                locationPoint.dateTime = dateTimeFrom(date: date)
            } else {
                locationPoint.dateTime = dateTimeFrom(date: Date())
            }
        } else {
            locationPoint.dateTime = dateTimeFrom(date: Date())
        }

        if let provider = json["provider"] as? String {
            locationPoint.provider = provider
        }

        if let isMockLocation = json["isMockLocation"] as? Bool {
            locationPoint.isMockLocation = isMockLocation
        }

        if let floor = json["floor"] as? Int32 {
            locationPoint.floorLevel = floor
        }

        if let batteryLevel = json["batteryLevel"] as? Int32 {
            locationPoint.batteryLevel = batteryLevel
        }

        if let networkType = json["networkType"] as? String {
            locationPoint.networkType = networkType
        }

        return locationPoint
    }

    private func parseActivityEvent(_ json: [String: Any])
        -> Chronotopia_ActivityEvent
    {
        var activityEvent = Chronotopia_ActivityEvent()

        // Parse activity type
        if let typeStr = json["type"] as? String {
            switch typeStr {
            case "in_vehicle":
                activityEvent.type = .inVehicle
            case "on_bicycle":
                activityEvent.type = .onBicycle
            case "on_foot":
                activityEvent.type = .onFoot
            case "running":
                activityEvent.type = .running
            case "still":
                activityEvent.type = .still
            case "tilting":
                activityEvent.type = .tilting
            case "walking":
                activityEvent.type = .walking
            default:
                activityEvent.type = .unknown
            }
        }

        // Parse confidence
        if let confidence = json["confidence"] as? Int32 {
            activityEvent.confidence = confidence
        }

        // Parse timestamps
        if let timestamp = json["timestamp"] as? Int64 {
            activityEvent.timestamp = timestamp
        } else {
            activityEvent.timestamp = Int64(Date().timeIntervalSince1970 * 1000)
        }

        if let startTime = json["startTime"] as? Int64 {
            activityEvent.startTime = startTime
        }

        if let endTime = json["endTime"] as? Int64 {
            activityEvent.endTime = endTime
        }

        // Parse movement data
        if let stepCount = json["stepCount"] as? Int64 {
            activityEvent.stepCount = stepCount
        }

        if let distance = json["distance"] as? Double {
            activityEvent.distance = distance
        }

        return activityEvent
    }

    private func startGRPCClient() {
        clientTask = Task {
            try await withGRPCClient(
                transport: .http2NIOPosix(
                    target: .dns(host: serverHost, port: serverPort),
                    transportSecurity: useSSL ? .tls : .plaintext,
                    config: .defaults { config in
                        // Customize the config
                        // config.backgroundActivityTimeout = .seconds(30)
                        config.backoff.initial = .milliseconds(100)
                        config.backoff.max = .seconds(5)
                    }
                )
            ) { [weak self] client in
                guard let self = self else { return }
                self.client = client
                self.tripTrackerClient = Chronotopia_Ingest.Client(
                    wrapping: client)

                // Process any pending data immediately
                await self.scheduleSyncData()

                // Keep the task running to maintain connections
                try await Task.sleep(for: .seconds(100_000_000))
            }
        }
    }

    private func sendDataPacket(
        _ dataPacket: Chronotopia_IngestBatch,
        using client: Chronotopia_Ingest.Client<HTTP2ClientTransport.Posix>
    ) async throws {
        // Use client streaming for sending data
        let response = try await client.submitBatch(dataPacket)

        print("Server response: \(response)")

        // Handle server instructions
        if response.pauseTracking {
            // Implement pause tracking logic if needed
            print("Server requested to pause tracking")
        }

        if response.recommendedUploadInterval > 0 {
            // Adjust upload frequency based on server recommendation
            print(
                "Server recommended upload interval: \(response.recommendedUploadInterval) seconds"
            )
            // TODO: Implement adjustment of upload frequency
        }
    }

    func shutdown() {
        // Cancel any sync task
        syncTask?.cancel()

        // Cancel client task
        clientTask?.cancel()

        // Clear references
        client = nil
        tripTrackerClient = nil
    }

    func now() -> Chronotopia_DateTime {
        // Get the current date and time
        let now = Date()

        return dateTimeFrom(date: now)

    }

    func dateTimeFrom(date: Date) -> Chronotopia_DateTime {
        // Get the current calendar and extract date components using the device's local time zone
        let calendar = Calendar.current
        let components = calendar.dateComponents(
            in: TimeZone.current, from: date)

        // Initialize Chronotopia_DateTime with current components
        var currentDateTime = Chronotopia_DateTime()
        currentDateTime.year = UInt32(components.year ?? 0)
        currentDateTime.month = UInt32(components.month ?? 0)
        currentDateTime.day = UInt32(components.day ?? 0)
        currentDateTime.hours = UInt32(components.hour ?? 0)
        currentDateTime.minutes = UInt32(components.minute ?? 0)
        currentDateTime.seconds = UInt32(components.second ?? 0)
        currentDateTime.nanos = UInt32(components.nanosecond ?? 0)

        // Determine the UTC offset in seconds
        let secondsFromGMT = TimeZone.current.secondsFromGMT(for: date)

        // Create a duration to represent the UTC offset and set the timeOffset field
        var utcOffsetDuration = SwiftProtobuf.Google_Protobuf_Duration()
        utcOffsetDuration.seconds = Int64(secondsFromGMT)
        currentDateTime.timeOffset = .utcOffset(utcOffsetDuration)
        return currentDateTime
    }
}

// Array extension to support chunking
extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}

// DispatchQueue extension for async/await
extension DispatchQueue {
    func async<T>(_ work: @escaping () -> T) async -> T {
        await withCheckedContinuation { continuation in
            self.async {
                let result = work()
                continuation.resume(returning: result)
            }
        }
    }
}
