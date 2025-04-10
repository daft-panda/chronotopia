import CoreLocation
import Flutter
import UIKit
import workmanager

@main
@objc class AppDelegate: FlutterAppDelegate {
    override func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication
            .LaunchOptionsKey: Any]?
    ) -> Bool {
        // Initialize Flutter engine
        let controller = window?.rootViewController as! FlutterViewController

        // Register method channels
        let serviceChannel = FlutterMethodChannel(
            name: "io.chronotopia.app/service",
            binaryMessenger: controller.binaryMessenger)
        let batteryChannel = FlutterMethodChannel(
            name: "io.chronotopia.app/battery",
            binaryMessenger: controller.binaryMessenger)
        let settingsChannel = FlutterMethodChannel(
            name: "io.chronotopia.app/settings",
            binaryMessenger: controller.binaryMessenger)

        // Register event channels
        _ = FlutterEventChannel(
            name: "io.chronotopia.app/location_stream",
            binaryMessenger: controller.binaryMessenger)
        _ = FlutterEventChannel(
            name: "io.chronotopia.app/activity_stream",
            binaryMessenger: controller.binaryMessenger)

        // Create LocationManager and MotionActivity instances
        let locationManager = LocationManager()
        let motionActivity = MotionActivity()

        // Register them with the appropriate channels
        LocationManager.register(
            with: controller.registrar(forPlugin: "LocationManager")!)
        MotionActivity.register(
            with: controller.registrar(forPlugin: "MotionActivity")!)

        // Create and register MethodChannelHandler
        let methodChannelHandler = MethodChannelHandler(
            locationManager: locationManager,
            motionActivity: motionActivity
        )

        // Set method call handlers
        serviceChannel.setMethodCallHandler(
            methodChannelHandler.handleServiceMethods)
        batteryChannel.setMethodCallHandler(
            methodChannelHandler.handleBatteryMethods)
        settingsChannel.setMethodCallHandler(
            methodChannelHandler.handleSettingsMethods)

        // Store references to prevent the objects from being deallocated
        objc_setAssociatedObject(
            self,
            &AssociatedObjectHandle.locationManager,
            locationManager,
            .OBJC_ASSOCIATION_RETAIN_NONATOMIC
        )

        objc_setAssociatedObject(
            self,
            &AssociatedObjectHandle.motionActivity,
            motionActivity,
            .OBJC_ASSOCIATION_RETAIN_NONATOMIC
        )

        objc_setAssociatedObject(
            self,
            &AssociatedObjectHandle.methodChannelHandler,
            methodChannelHandler,
            .OBJC_ASSOCIATION_RETAIN_NONATOMIC
        )

        WorkmanagerPlugin.registerPeriodicTask(
            withIdentifier: "io.chronotopia.app.locationTrackingTask",
            frequency: NSNumber(value: 20 * 60)
        )
        WorkmanagerPlugin.registerPeriodicTask(
            withIdentifier: "io.chronotopia.app.dataUploadTask",
            frequency: NSNumber(value: 20 * 60)
        )
        WorkmanagerPlugin.registerPeriodicTask(
            withIdentifier: "io.chronotopia.app.serviceRestartTask",
            frequency: NSNumber(value: 20 * 60)
        )

        GeneratedPluginRegistrant.register(with: self)
        return super.application(
            application, didFinishLaunchingWithOptions: launchOptions)
    }
}

// Used for associated object handles
private struct AssociatedObjectHandle {
    static var locationManager: UInt8 = 0
    static var motionActivity: UInt8 = 0
    static var methodChannelHandler: UInt8 = 0
}
