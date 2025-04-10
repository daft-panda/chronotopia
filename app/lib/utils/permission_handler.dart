import 'package:permission_handler/permission_handler.dart';

class PermissionHandler {
  Future<void> requestRequiredPermissions() async {
    final locationStatus = await Permission.locationAlways.request();
    final activityStatus = await Permission.activityRecognition.request();

    if (locationStatus.isDenied || activityStatus.isDenied) {
      // Handle denied permissions
      // Consider showing dialog explaining why permissions are needed
    }

    // iOS specific permissions
    if (await Permission.locationAlways.isPermanentlyDenied) {
      // Guide users to app settings
      openAppSettings();
    }
  }
}
