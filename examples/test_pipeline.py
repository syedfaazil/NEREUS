"""
Example script to test the Zooplankton pipeline
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from pipeline import ZooplanktonPipeline


def test_single_image():
    """Test pipeline on a single image"""
    print("=" * 50)
    print("Testing Zooplankton Pipeline - Single Image")
    print("=" * 50)
    
    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = ZooplanktonPipeline(
        num_classes=127,
        device='cpu',
        enable_preprocessing=True,
        enable_watershed=True
    )
    
    # Print pipeline info
    info = pipeline.get_pipeline_info()
    print(f"\nPipeline Configuration:")
    print(f"  Device: {info['device']}")
    print(f"  Preprocessing: {info['preprocessing_enabled']}")
    print(f"  Watershed: {info['watershed_enabled']}")
    
    # Load test image
    print("\n2. Loading test image...")
    # Try to load an image from the dataset
    image_dir = Path(__file__).parent.parent / 'individual_images'
    
    # Find first available image
    test_image_path = None
    for subdir in image_dir.iterdir():
        if subdir.is_dir():
            images = list(subdir.glob('*.jpg')) + list(subdir.glob('*.png'))
            if images:
                test_image_path = images[0]
                break
    
    if test_image_path is None:
        print("No test images found. Creating synthetic image...")
        # Create a synthetic test image
        image = np.ones((640, 640), dtype=np.uint8) * 200
        # Add some circular objects
        cv2.circle(image, (200, 200), 50, 100, -1)
        cv2.circle(image, (400, 400), 60, 120, -1)
        cv2.circle(image, (300, 500), 40, 80, -1)
    else:
        print(f"Loading image: {test_image_path}")
        image = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)
    
    print(f"Image shape: {image.shape}")
    
    # Process image
    print("\n3. Processing image through pipeline...")
    result = pipeline.process_image(image, return_visualization=True)
    
    # Display results
    print(f"\n4. Results:")
    print(f"  Number of detections: {result['num_detections']}")
    print(f"  Processing time: {result['processing_time']:.3f} seconds")
    
    if result['detections']:
        print(f"\n  Detections:")
        for i, det in enumerate(result['detections'], 1):
            print(f"    Detection {i}:")
            print(f"      BBox: {det['bbox']}")
            print(f"      Confidence: {det['confidence']:.3f}")
            if det['top_species']:
                species_id, conf = det['top_species']
                print(f"      Top Species: ID={species_id}, Conf={conf:.3f}")
    
    # Get statistics
    stats = pipeline.get_statistics()
    print(f"\n5. Pipeline Statistics:")
    print(f"  Total images processed: {stats['total_images_processed']}")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Average FPS: {stats['fps']:.2f}")
    
    # Save visualization if available
    if 'visualization' in result:
        output_path = Path(__file__).parent / 'output_visualization.jpg'
        cv2.imwrite(str(output_path), result['visualization'])
        print(f"\n6. Visualization saved to: {output_path}")
    
    print("\n" + "=" * 50)
    print("Test completed successfully!")
    print("=" * 50)


def test_preprocessing():
    """Test preprocessing modules independently"""
    print("\n" + "=" * 50)
    print("Testing Preprocessing Modules")
    print("=" * 50)
    
    from preprocessing import ImageEnhancer, ArtifactRemover, WatershedSegmenter
    
    # Create test image
    image = np.ones((640, 640), dtype=np.uint8) * 150
    cv2.circle(image, (200, 200), 50, 100, -1)
    cv2.circle(image, (400, 400), 60, 120, -1)
    
    print("\n1. Testing Image Enhancement...")
    enhancer = ImageEnhancer()
    enhanced = enhancer.adaptive_enhancement(image)
    print(f"   Original mean: {np.mean(image):.2f}")
    print(f"   Enhanced mean: {np.mean(enhanced):.2f}")
    
    print("\n2. Testing Artifact Removal...")
    artifact_remover = ArtifactRemover()
    cleaned = artifact_remover.remove_artifacts(enhanced)
    print(f"   Artifacts removed successfully")
    
    print("\n3. Testing Watershed Segmentation...")
    watershed = WatershedSegmenter()
    count = watershed.count_objects(cleaned)
    print(f"   Objects detected: {count}")
    
    print("\nPreprocessing tests completed!")


def test_detection():
    """Test detection module independently"""
    print("\n" + "=" * 50)
    print("Testing Detection Module")
    print("=" * 50)
    
    from detection import YOLODetector
    
    print("\n1. Initializing YOLOv5n detector...")
    detector = YOLODetector(device='cpu')
    
    info = detector.get_model_info()
    print(f"\nDetector Info:")
    print(f"  Model: {info['model_type']}")
    print(f"  Parameters: {info['parameters']:,}")
    print(f"  Device: {info['device']}")
    
    # Create test image
    image = np.ones((640, 640, 3), dtype=np.uint8) * 200
    cv2.circle(image, (200, 200), 50, (100, 100, 100), -1)
    
    print("\n2. Running detection...")
    detections, _ = detector.detect(image)
    print(f"   Detections: {len(detections)}")
    
    print("\nDetection test completed!")


def test_classification():
    """Test classification module independently"""
    print("\n" + "=" * 50)
    print("Testing Classification Module")
    print("=" * 50)
    
    from classification import MobileNetClassifier
    
    print("\n1. Initializing MobileNetV3 classifier...")
    classifier = MobileNetClassifier(num_classes=127, device='cpu')
    
    info = classifier.get_model_info()
    print(f"\nClassifier Info:")
    print(f"  Model: {info['model_type']}")
    print(f"  Classes: {info['num_classes']}")
    print(f"  Parameters: {info['total_parameters']:,}")
    print(f"  Model size: {info['model_size_mb']:.2f} MB")
    
    # Create test image
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    print("\n2. Running classification...")
    predictions = classifier.classify(image, top_k=5)
    print(f"   Top 5 predictions:")
    for i, (class_id, conf) in enumerate(predictions, 1):
        print(f"     {i}. Class {class_id}: {conf:.3f}")
    
    print("\n3. Extracting embeddings...")
    embeddings = classifier.extract_embeddings(image)
    print(f"   Embedding shape: {embeddings.shape}")
    
    print("\nClassification test completed!")


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("Zooplankton Pipeline Test Suite")
    print("=" * 50)
    
    # Run all tests
    try:
        test_preprocessing()
        test_detection()
        test_classification()
        test_single_image()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
