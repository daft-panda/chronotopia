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
    private var serverHost: String
    private var serverPort = 10000
    private var useSSL = true

    // Authorization token
    private var authToken: String?

    // Status tracking
    private var lastSuccessfulUploadTime: Date?
    private var lastUploadError: String?
    private var uploadAttemptCount: Int = 0
    private var uploadSuccessCount: Int = 0

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
        serverHost =
            defaults.string(forKey: "server_url") ?? "ingest.chronotopia.io"
        useSSL = defaults.bool(forKey: "use_tls")

        // Load authorization token if available
        authToken = defaults.string(forKey: "auth_token")

        // Load previous status if available
        if let lastUploadTimeInterval = defaults.object(
            forKey: "last_upload_time") as? TimeInterval
        {
            lastSuccessfulUploadTime = Date(
                timeIntervalSince1970: lastUploadTimeInterval)
        }
        lastUploadError = defaults.string(forKey: "last_upload_error")
        uploadAttemptCount = defaults.integer(forKey: "upload_attempt_count")
        uploadSuccessCount = defaults.integer(forKey: "upload_success_count")

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

    // Get status for Flutter UI
    func getUploadStatus() -> [String: Any] {
        var status: [String: Any] = [
            "hasNetworkConnection": hasNetworkConnection,
            "uploadAttemptCount": uploadAttemptCount,
            "uploadSuccessCount": uploadSuccessCount,
            "isSyncing": isSyncing,
        ]

        if let lastSuccessfulUploadTime = lastSuccessfulUploadTime {
            status["lastUploadTime"] =
                lastSuccessfulUploadTime.timeIntervalSince1970
        }

        if let lastUploadError = lastUploadError {
            status["lastError"] = lastUploadError
        }

        // Count pending files
        do {
            let fileManager = FileManager.default
            let locationFiles = try fileManager.contentsOfDirectory(
                at: locationCacheURL, includingPropertiesForKeys: nil)
            let activityFiles = try fileManager.contentsOfDirectory(
                at: activityCacheURL, includingPropertiesForKeys: nil)

            status["pendingLocationCount"] = locationFiles.count
            status["pendingActivityCount"] = activityFiles.count
        } catch {
            print("Error counting pending files: \(error)")
        }

        return status
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

    // Set the authorization token
    func setAuthToken(_ token: String) {
        queue.async {
            self.authToken = token

            // Save to UserDefaults
            let defaults = UserDefaults.standard
            defaults.set(token, forKey: "auth_token")
            defaults.synchronize()

            // Restart the client to use the new token
            self.restartClient()
        }
    }

    // Set the server settings
    func setServerSettings(serverUrl: String, useTLS: Bool) {
        queue.async {
            // If settings changed, update and restart client
            let settingsChanged =
                self.serverHost != serverUrl || self.useSSL != useTLS

            // Update settings
            self.serverHost = serverUrl
            self.useSSL = useTLS

            // Save to UserDefaults
            let defaults = UserDefaults.standard
            defaults.set(serverUrl, forKey: "server_url")
            defaults.set(useTLS, forKey: "use_tls")
            defaults.synchronize()

            // Restart the client if settings changed
            if settingsChanged {
                self.restartClient()
            }
        }
    }

    // Restart the gRPC client
    private func restartClient() {
        // Cancel any existing client
        clientTask?.cancel()
    }

    private func initDeviceMetadata() {
        deviceMetadata = Chronotopia_DeviceMetadata()
        deviceMetadata?.platform = "ios"
        deviceMetadata?.osVersion = UIDevice.current.systemVersion
        deviceMetadata?.deviceModel = UIDevice.current.model
        deviceMetadata?.deviceLanguage =
            Locale.current.language.languageCode?.identifier ?? "en"

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
                    self.updateUploadError(error.localizedDescription)
                }
            }
        }
    }

    private func updateUploadError(_ errorMessage: String) {
        self.lastUploadError = errorMessage
        let defaults = UserDefaults.standard
        defaults.set(errorMessage, forKey: "last_upload_error")
        defaults.synchronize()
    }

    private func recordSuccessfulUpload() {
        // Update timestamps and counters
        self.lastSuccessfulUploadTime = Date()
        self.uploadSuccessCount += 1
        self.lastUploadError = nil

        // Save to UserDefaults
        let defaults = UserDefaults.standard
        defaults.set(
            self.lastSuccessfulUploadTime?.timeIntervalSince1970,
            forKey: "last_upload_time")
        defaults.set(self.uploadSuccessCount, forKey: "upload_success_count")
        defaults.set(nil, forKey: "last_upload_error")
        defaults.synchronize()
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
            updateUploadError("No network connection")
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

            // Update attempt counter
            await queue.async {
                self.uploadAttemptCount += 1
                let defaults = UserDefaults.standard
                defaults.set(
                    self.uploadAttemptCount, forKey: "upload_attempt_count")
                defaults.synchronize()
            }

            // Process location batches
            for batch in locationBatches {
                // Create data packet
                var dataPacket = Chronotopia_IngestBatch()
                dataPacket.dateTime = now()

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
                        updateUploadError(
                            "Error processing location file: \(error.localizedDescription)"
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
                    withIngestClient { ingestClient in
                        try await self.sendDataPacket(
                            dataPacket, using: ingestClient)
                    }

                    // Record successful upload
                    recordSuccessfulUpload()
                }
            }

            // Process activity batches
            for batch in activityBatches {
                // Create data packet
                var ingestBatch = Chronotopia_IngestBatch()
                ingestBatch.dateTime = now()

                // Process each file in batch
                for fileURL in batch {
                    do {
                        let jsonData = try Data(contentsOf: fileURL)
                        let jsonObject =
                            try JSONSerialization.jsonObject(with: jsonData)
                            as! [String: Any]

                        let activityEvent = parseActivityEvent(jsonObject)
                        ingestBatch.activities.append(activityEvent)

                        // Delete file after successful processing
                        try fileManager.removeItem(at: fileURL)
                    } catch {
                        print(
                            "Error processing activity file \(fileURL.lastPathComponent): \(error)"
                        )
                        updateUploadError(
                            "Error processing activity file: \(error.localizedDescription)"
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
                if !ingestBatch.activities.isEmpty {
                    withIngestClient {
                        ingestClient in
                        try await self.sendDataPacket(
                            ingestBatch, using: ingestClient)
                    }

                    // Record successful upload
                    recordSuccessfulUpload()
                }
            }

        } catch {
            print("Error during data sync: \(error)")
            updateUploadError("Sync error: \(error.localizedDescription)")
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

        if let isMockLocation = json["isMockLocation"] as? Bool {
            locationPoint.isMockLocation = isMockLocation
        }

        if let floor = json["floor"] as? Int32 {
            locationPoint.floorLevel = floor
        }

        if let batteryLevel = json["batteryLevel"] as? Int32 {
            if batteryLevel != -1 {
                locationPoint.batteryLevel = UInt32(batteryLevel)
            }
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
        if let startTimeStr = json["startTime"] as? String {
            let formatter = ISO8601DateFormatter()
            if let date = formatter.date(from: startTimeStr) {
                activityEvent.start = dateTimeFrom(date: date)
            }
        }

        if let endTimeStr = json["endTime"] as? String {
            let formatter = ISO8601DateFormatter()
            if let date = formatter.date(from: endTimeStr) {
                activityEvent.end = dateTimeFrom(date: date)
            }
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

    private func withIngestClient(
        operation: @escaping (
            Chronotopia_Ingest.Client<HTTP2ClientTransport.Posix>
        ) async throws ->
            Void
    ) {
        print("gRPC to host: \(serverHost), SSL: \(useSSL)")

        clientTask = Task {
            do {
                try await withGRPCClient(
                    transport: .http2NIOPosix(
                        target: .dns(host: serverHost, port: serverPort),
                        transportSecurity: useSSL ? .tls : .plaintext,
                        config: .defaults { config in
                            // Customize the config
                            config.backoff.initial = .milliseconds(100)
                            config.backoff.max = .seconds(5)
                        }
                    )
                ) { [weak self] client in
                    guard self != nil else { return }

                    // Create the Ingest client
                    let ingestClient = Chronotopia_Ingest.Client(
                        wrapping: client)

                    // Execute the provided operation with the client
                    try await operation(ingestClient)
                }
            } catch {
                print("gRPC client error: \(error)")
                self.updateUploadError(
                    "Connection error: \(error.localizedDescription)")
            }
        }
    }

    private func sendDataPacket(
        _ dataPacket: Chronotopia_IngestBatch,
        using client: Chronotopia_Ingest.Client<HTTP2ClientTransport.Posix>
    ) async throws {
        // Create metadata with authorization token
        var metadata = GRPCCore.Metadata()
        if let authToken = authToken {
            metadata.addString("Bearer \(authToken)", forKey: "authorization")
        }

        // Use client streaming for sending data with metadata
        let response = try await client.submitBatch(
            dataPacket,
            metadata: metadata
        )

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

    // Submit device metadata to the server
    func submitDeviceMetadata() async throws {
        guard let deviceMetadata = deviceMetadata else {
            print("Device metadata or client not initialized")
            return
        }

        // Create metadata with authorization token
        var metadata = GRPCCore.Metadata()
        if let authToken = authToken {
            metadata.addString("Bearer \(authToken)", forKey: "authorization")
        }

        withIngestClient {
            ingestClient in
            // Submit device metadata
            let response = try await ingestClient.submitDeviceMetadata(
                deviceMetadata,
                metadata: metadata
            )

            print("Device metadata submission response: \(response)")
        }
    }

    func shutdown() {
        // Cancel any sync task
        syncTask?.cancel()

        // Cancel client task
        clientTask?.cancel()
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
