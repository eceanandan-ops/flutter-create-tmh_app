// lib/main.dart
import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as imgpkg;
import 'package:permission_handler/permission_handler.dart';
import 'package:path_provider/path_provider.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    cameras = await availableCameras();
  } catch (e) {
    // ignore
  }
  runApp(const TMHApp());
}

class TMHApp extends StatelessWidget {
  const TMHApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'TMH Measurement',
      theme: ThemeData(primarySwatch: Colors.indigo),
      home: const CameraPage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class CameraPage extends StatefulWidget {
  const CameraPage({super.key});
  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  CameraController? _controller;
  bool _isDetecting = false;
  double _tmhPixels = 0.0;
  double _tmhMm = 0.0;
  bool _cameraInitialized = false;
  bool _calibrated = false;
  double _pixelsPerMm = 0.0; // calibration factor: pixels per mm
  final faceDetector = FaceDetector(
      options: FaceDetectorOptions(
    enableContours: true,
    enableLandmarks: true,
    performanceMode: FaceDetectorMode.fast,
  ));

  @override
  void initState() {
    super.initState();
    _initPermissionsAndCamera();
  }

  Future<void> _initPermissionsAndCamera() async {
    final camStatus = await Permission.camera.request();
    if (!camStatus.isGranted) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Camera permission is required')));
      return;
    }

    if (cameras.isEmpty) {
      if (!mounted) return;
      ScaffoldMessenger.of(context)
          .showSnackBar(const SnackBar(content: Text('No camera found')));
      return;
    }

    // Choose front or back camera — usually back camera provides better focus for eye capture,
    // but users may prefer front. We'll choose back camera if available.
    CameraDescription chosen = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first);

    _controller = CameraController(chosen, ResolutionPreset.medium,
        imageFormatGroup: ImageFormatGroup.yuv420);
    try {
      await _controller!.initialize();
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('Camera init error: $e')));
      return;
    }

    setState(() {
      _cameraInitialized = true;
    });

    _controller!.startImageStream(_processCameraImage);
  }

  // process frames from camera
  void _processCameraImage(CameraImage image) async {
    if (_isDetecting) return;
    _isDetecting = true;

    try {
      final inputImage = _convertCameraImage(image, _controller!.description);

      final faces = await faceDetector.processImage(inputImage);
      if (faces.isNotEmpty) {
        final face = faces.first;
        // use eye landmarks or contours if available
        // try to determine lower eyelid ROI using eye landmarks or contour points
        Rect? eyeRect = _estimateEyeRect(face);
        if (eyeRect != null) {
          // convert CameraImage YUV to Dart image, crop ROI, analyze
          imgpkg.Image? converted = await compute(_convertYUV420ToImage, image);
          if (converted != null) {
            // Crop to eyeRect with scaling from inputImage to converted image size
            final scaleX = converted.width / inputImage.inputImageData!.size.width;
            final scaleY = converted.height / inputImage.inputImageData!.size.height;
            final cropRect = Rect.fromLTWH(
              (eyeRect.left * scaleX).clamp(0.0, converted.width - 1).toDouble(),
              (eyeRect.top * scaleY).clamp(0.0, converted.height - 1).toDouble(),
              (eyeRect.width * scaleX).clamp(1.0, converted.width.toDouble()).toDouble(),
              (eyeRect.height * scaleY).clamp(1.0, converted.height.toDouble()).toDouble(),
            );
            final cropped = imgpkg.copyCrop(
                converted,
                cropRect.left.toInt(),
                cropRect.top.toInt(),
                cropRect.width.toInt(),
                cropRect.height.toInt());

            // Analyze cropped image to find tear meniscus height in pixels
            final double tmhPixels = await compute(_estimateTMHFromEyeImage, cropped);

            // update conversion to mm if calibrated
            double tmhMm = _calibrated && _pixelsPerMm > 0
                ? tmhPixels / _pixelsPerMm
                : _estimateUsingDefaultConversion(tmhPixels, cropRect.width);

            setState(() {
              _tmhPixels = tmhPixels;
              _tmhMm = double.parse(tmhMm.toStringAsFixed(2));
            });
          }
        }
      }
    } catch (e) {
      // ignore processing errors
    } finally {
      _isDetecting = false;
    }
  }

  // Convert CameraImage to ML Kit InputImage wrapper
  InputImage _convertCameraImage(CameraImage image, CameraDescription desc) {
    final allBytes = WriteBuffer();
    for (final plane in image.planes) {
      allBytes.putUint8List(plane.bytes);
    }
    final bytes = allBytes.done().buffer.asUint8List();

    final Size imageSize = Size(image.width.toDouble(), image.height.toDouble());
    final imageRotation = InputImageRotationValue.fromRawValue(
            _rotationIntToImageRotation(desc.sensorOrientation).rawValue) ??
        InputImageRotation.rotation0deg;

    final inputImageFormat =
        InputImageFormatValue.fromRawValue(image.format.group.index) ??
            InputImageFormat.nv21;

    final planeData = image.planes.map(
      (plane) {
        return InputImagePlaneMetadata(
          bytesPerRow: plane.bytesPerRow,
          height: plane.height,
          width: plane.width,
        );
      },
    ).toList();

    final data = InputImageData(
      size: imageSize,
      imageRotation: imageRotation,
      inputImageFormat: inputImageFormat,
      planeData: planeData,
    );

    return InputImage.fromBytes(bytes: bytes, inputImageData: data);
  }

  // helper to convert rotation ints
  InputImageRotation _rotationIntToImageRotation(int rotation) {
    switch (rotation) {
      case 0:
        return InputImageRotation.rotation0deg;
      case 90:
        return InputImageRotation.rotation90deg;
      case 180:
        return InputImageRotation.rotation180deg;
      default:
        return InputImageRotation.rotation270deg;
    }
  }

  // Estimate rectangle for eye region using landmarks/contours
  Rect? _estimateEyeRect(Face face) {
    // Prefer left or right eye landmarks. We'll try both and take one with larger area.
    final l = face.getLandmark(FaceLandmarkType.leftEye);
    final r = face.getLandmark(FaceLandmarkType.rightEye);

    late Offset center;
    double boxW = 0;
    double boxH = 0;

    if (l != null) {
      center = Offset(l.position.x, l.position.y);
      boxW = face.boundingBox.width * 0.28;
      boxH = face.boundingBox.height * 0.14;
      final rect = Rect.fromCenter(center: center, width: boxW, height: boxH);
      return rect;
    } else if (r != null) {
      center = Offset(r.position.x, r.position.y);
      boxW = face.boundingBox.width * 0.28;
      boxH = face.boundingBox.height * 0.14;
      final rect = Rect.fromCenter(center: center, width: boxW, height: boxH);
      return rect;
    } else {
      // fallback: use part of face bounding box near lower area
      final bb = face.boundingBox;
      return Rect.fromLTWH(bb.left + bb.width * 0.15, bb.top + bb.height * 0.45,
          bb.width * 0.7, bb.height * 0.25);
    }
  }

  // Basic fallback conversion: assume corneal horizontal diameter ~11.7 mm
  double _estimateUsingDefaultConversion(double tmhPixels, double cropWidthPx) {
    // assume corneal diameter in mm and approximate pixels-per-mm by equating crop width to cornea width
    const double corneaMm = 11.7; // average horizontal corneal diameter (mm)
    // If cropWidthPx is zero, return 0
    if (cropWidthPx <= 0) return 0.0;
    final double pxPerMm = cropWidthPx / corneaMm;
    return tmhPixels / pxPerMm;
  }

  // User-initiated calibration: provide known length (mm) and measure pixel length on screen
  Future<void> _calibrate() async {
    // For simplicity: ask user to take a photo with a small ruler/sticker of known width placed next to eye
    // We'll capture a frame and let the processing estimate a horizontal object by asking user to align.
    if (_controller == null || !_controller!.value.isInitialized) return;
    try {
      final XFile file = await _controller!.takePicture();
      final bytes = await file.readAsBytes();
      final img = imgpkg.decodeImage(bytes);
      if (img == null) return;

      // For this demo, we'll assume user placed a calibration sticker across the eye and we take full width
      // Ask the user for real-world length in mm via a dialog
      final mmStr = await _askForInput(context,
          title: 'Calibration', message: 'Enter known object length in mm (e.g., 10):');
      if (mmStr == null) return;
      final knownMm = double.tryParse(mmStr);
      if (knownMm == null || knownMm <= 0) return;

      // crude: estimate object pixel width by taking width of image; in real app, let the user draw a line
      final pxWidth = img.width.toDouble();
      final pxPerMm = pxWidth / knownMm;

      setState(() {
        _pixelsPerMm = pxPerMm;
        _calibrated = true;
      });

      ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Calibration done. pixels/mm = ${pxPerMm.toStringAsFixed(2)}')));
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Calibration error: $e')));
      }
    }
  }

  Future<String?> _askForInput(BuildContext ctx,
      {required String title, required String message}) {
    final TextEditingController c = TextEditingController();
    return showDialog<String>(
      context: ctx,
      builder: (context) {
        return AlertDialog(
          title: Text(title),
          content: Column(mainAxisSize: MainAxisSize.min, children: [
            Text(message),
            TextField(
              controller: c,
              keyboardType: TextInputType.numberWithOptions(decimal: true),
            )
          ]),
          actions: [
            TextButton(onPressed: () => Navigator.pop(context, null), child: const Text('Cancel')),
            TextButton(onPressed: () => Navigator.pop(context, c.text), child: const Text('OK'))
          ],
        );
      },
    );
  }

  // Flush detector
  @override
  void dispose() {
    _controller?.dispose();
    faceDetector.close();
    super.dispose();
  }

  // UI
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('TMH Measurement (Demo)'),
        actions: [
          IconButton(
              icon: const Icon(Icons.build),
              tooltip: 'Calibrate (place a ruler near the eye and follow prompts)',
              onPressed: _calibrate)
        ],
      ),
      body: SafeArea(
        child: Column(children: [
          Expanded(
            child: _cameraInitialized && _controller != null
                ? CameraPreview(_controller!)
                : const Center(child: CircularProgressIndicator()),
          ),
          Container(
            color: Colors.grey[100],
            padding: const EdgeInsets.all(12),
            child: Row(
              children: [
                Expanded(
                    child: Text('TMH: ${_tmhMm.toStringAsFixed(2)} mm '
                        '(${_tmhPixels.toStringAsFixed(1)} px)',
                        style: const TextStyle(fontSize: 18))),
                ElevatedButton.icon(
                    onPressed: _captureAndSaveImage,
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Capture')),
              ],
            ),
          ),
        ]),
      ),
    );
  }

  Future<void> _captureAndSaveImage() async {
    if (_controller == null) return;
    final XFile file = await _controller!.takePicture();
    final dir = await getApplicationDocumentsDirectory();
    final newPath = '${dir.path}/tmh_capture_${DateTime.now().millisecondsSinceEpoch}.jpg';
    await file.saveTo(newPath);
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Saved: $newPath')));
    }
  }
}

/// Worker function: convert CameraImage (YUV420) to image package Image
/// We run this in an isolate using compute()
Future<imgpkg.Image?> _convertYUV420ToImage(CameraImage cameraImage) async {
  try {
    final int width = cameraImage.width;
    final int height = cameraImage.height;

    final img = imgpkg.Image(width, height); // RGB image

    final Plane p0 = cameraImage.planes[0];
    final Plane p1 = cameraImage.planes[1];
    final Plane p2 = cameraImage.planes[2];

    final Uint8List y = p0.bytes;
    final Uint8List u = p1.bytes;
    final Uint8List v = p2.bytes;

    final int uvRowStride = p1.bytesPerRow;
    final int uvPixelStride = p1.bytesPerPixel ?? 1;

    int yp = 0;
    for (int j = 0; j < height; j++) {
      final int uvRow = (j / 2).floor();
      for (int i = 0; i < width; i++) {
        final int uvCol = (i / 2).floor();
        final int indexU = uvRow * uvRowStride + uvCol * uvPixelStride;
        final int indexV = indexU; // likely interleaved
        final int Y = y[yp];
        final int U = u[indexU];
        final int V = v[indexV];

        int r = (Y + (1.370705 * (V - 128))).round();
        int g = (Y - (0.337633 * (U - 128)) - (0.698001 * (V - 128))).round();
        int b = (Y + (1.732446 * (U - 128))).round();

        r = r.clamp(0, 255);
        g = g.clamp(0, 255);
        b = b.clamp(0, 255);

        img.setPixelRgba(i, j, r, g, b);

        yp++;
      }
    }

    return img;
  } catch (e) {
    return null;
  }
}

/// Worker: estimate TMH in pixels from a cropped eye image
/// strategy (simplified):
/// 1. Convert to grayscale
/// 2. Apply Gaussian blur
/// 3. Use vertical intensity projection in lower half of the eye crop to find the dark band (tear meniscus)
/// 4. Estimate height in pixels
double _estimateTMHFromEyeImage(imgpkg.Image eyeImage) {
  // convert to grayscale
  final gray = imgpkg.grayscale(eyeImage);

  // take lower half of the crop where tear meniscus usually lies
  int startY = (gray.height * 0.45).floor();
  int h = gray.height - startY;
  if (h <= 2) h = (gray.height / 2).floor();

  // compute vertical projection of darkness: for each row compute mean intensity
  List<double> rowMeans = List.filled(h, 0.0);
  for (int y = 0; y < h; y++) {
    int yy = startY + y;
    double sum = 0;
    for (int x = 0; x < gray.width; x++) {
      final p = gray.getPixel(x, yy);
      final intensity = imgpkg.getLuminance(p).toDouble(); // 0..255
      sum += intensity;
    }
    rowMeans[y] = sum / gray.width;
  }

  // We expect a dark band (lower intensity) for tear meniscus. Normalize and find contiguous dark region.
  final double meanAll = rowMeans.reduce((a, b) => a + b) / rowMeans.length;
  final double threshold = meanAll * 0.92; // heuristic threshold (adjustable)
  int topBound = -1;
  int bottomBound = -1;

  for (int i = 0; i < rowMeans.length; i++) {
    if (rowMeans[i] < threshold) {
      if (topBound == -1) topBound = i;
      bottomBound = i;
    } else {
      // once passed and we already had region, break
      if (topBound != -1) break;
    }
  }

  if (topBound == -1 || bottomBound == -1) {
    // fallback: return small value
    return max(0.0, gray.height * 0.02);
  }

  final double tmhPixels = (bottomBound - topBound + 1).toDouble();
  return tmhPixels;
}
