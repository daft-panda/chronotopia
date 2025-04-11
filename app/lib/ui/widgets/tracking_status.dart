// widgets/tracking_status_card.dart
import 'dart:async';
import 'package:chronotopia_app/services/background_service.dart';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';

class TrackingStatusCard extends StatefulWidget {
  final bool isTracking;
  final Function(bool) onToggle;

  const TrackingStatusCard({
    super.key,
    required this.isTracking,
    required this.onToggle,
  });

  @override
  State<TrackingStatusCard> createState() => _TrackingStatusCardState();
}

class _TrackingStatusCardState extends State<TrackingStatusCard> {
  Map<String, dynamic> _uploadStatus = {};
  Timer? _refreshTimer;
  bool _isExpanded = false;

  @override
  void initState() {
    super.initState();
    _fetchUploadStatus();

    // Refresh status every 10 seconds
    _refreshTimer = Timer.periodic(const Duration(seconds: 10), (_) {
      _fetchUploadStatus();
    });
  }

  @override
  void dispose() {
    _refreshTimer?.cancel();
    super.dispose();
  }

  Future<void> _fetchUploadStatus() async {
    final backgroundService = Provider.of<BackgroundService>(
      context,
      listen: false,
    );
    final status = await backgroundService.getUploadStatus();

    if (mounted) {
      setState(() {
        _uploadStatus = status;
      });
    }
  }

  String _formatDateTime(double? timestamp) {
    if (timestamp == null) return 'Never';

    final date = DateTime.fromMillisecondsSinceEpoch(
      (timestamp * 1000).toInt(),
    );
    return DateFormat('MMM d, h:mm a').format(date);
  }

  @override
  Widget build(BuildContext context) {
    final hasNetworkConnection =
        _uploadStatus['hasNetworkConnection'] as bool? ?? false;
    final isSyncing = _uploadStatus['isSyncing'] as bool? ?? false;
    final lastUploadTime = _uploadStatus['lastUploadTime'] as double?;
    final lastError = _uploadStatus['lastError'] as String?;
    final pendingLocationCount =
        _uploadStatus['pendingLocationCount'] as int? ?? 0;
    final pendingActivityCount =
        _uploadStatus['pendingActivityCount'] as int? ?? 0;
    final uploadAttemptCount = _uploadStatus['uploadAttemptCount'] as int? ?? 0;
    final uploadSuccessCount = _uploadStatus['uploadSuccessCount'] as int? ?? 0;

    final pendingCount = pendingLocationCount + pendingActivityCount;
    final lastUploadText =
        lastUploadTime != null
            ? 'Last upload: ${_formatDateTime(lastUploadTime)}'
            : 'No uploads yet';

    return Card(
      color: widget.isTracking ? Colors.green.shade50 : Colors.red.shade50,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'Location Tracking',
                  style: TextStyle(fontSize: 18.0, fontWeight: FontWeight.bold),
                ),
                Switch(
                  value: widget.isTracking,
                  onChanged: widget.onToggle,
                  activeColor: Colors.green,
                ),
              ],
            ),
            const SizedBox(height: 8.0),
            Text(
              widget.isTracking
                  ? 'Tracking is active. Your timeline is being built.'
                  : 'Tracking is paused. No data is being collected.',
              style: TextStyle(
                color:
                    widget.isTracking
                        ? Colors.green.shade700
                        : Colors.red.shade700,
              ),
            ),

            // Network status indicator
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 8.0),
              child: Row(
                children: [
                  Icon(
                    hasNetworkConnection ? Icons.wifi : Icons.wifi_off,
                    color: hasNetworkConnection ? Colors.green : Colors.red,
                    size: 16,
                  ),
                  const SizedBox(width: 8),
                  Text(
                    hasNetworkConnection ? 'Connected' : 'No connection',
                    style: TextStyle(
                      color:
                          hasNetworkConnection
                              ? Colors.green.shade700
                              : Colors.red.shade700,
                    ),
                  ),
                  if (isSyncing) ...[
                    const SizedBox(width: 8),
                    const SizedBox(
                      width: 12,
                      height: 12,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                    const SizedBox(width: 4),
                    const Text('Syncing...'),
                  ],
                ],
              ),
            ),

            // Upload status
            Text(lastUploadText, style: const TextStyle(fontSize: 12.0)),

            // Pending data info
            if (pendingCount > 0)
              Text(
                'Pending uploads: $pendingCount items',
                style: TextStyle(
                  fontSize: 12.0,
                  color:
                      pendingCount > 100
                          ? Colors.orange.shade800
                          : Colors.grey.shade700,
                ),
              ),

            // Error display
            if (lastError != null && lastError.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 8.0),
                child: Text(
                  'Error: $lastError',
                  style: TextStyle(fontSize: 12.0, color: Colors.red.shade800),
                ),
              ),

            // Expander for more details
            InkWell(
              onTap: () {
                setState(() {
                  _isExpanded = !_isExpanded;
                });
              },
              child: Padding(
                padding: const EdgeInsets.symmetric(vertical: 8.0),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      _isExpanded ? Icons.expand_less : Icons.expand_more,
                      size: 16,
                    ),
                    const SizedBox(width: 4),
                    Text(
                      _isExpanded ? 'Hide details' : 'Show details',
                      style: const TextStyle(fontSize: 12),
                    ),
                  ],
                ),
              ),
            ),

            // Expanded details
            if (_isExpanded) ...[
              const Divider(),
              Text('Upload attempts: $uploadAttemptCount'),
              Text('Successful uploads: $uploadSuccessCount'),
              Text('Pending locations: $pendingLocationCount'),
              Text('Pending activities: $pendingActivityCount'),

              // Force sync button
              Padding(
                padding: const EdgeInsets.only(top: 8.0),
                child: ElevatedButton.icon(
                  onPressed: () async {
                    final service = Provider.of<BackgroundService>(
                      context,
                      listen: false,
                    );
                    await service.forceSyncNow();
                    _fetchUploadStatus();
                  },
                  icon: const Icon(Icons.sync),
                  label: const Text('Force Sync'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 8,
                    ),
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
