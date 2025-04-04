#include <Arduino.h>
#include <SPI.h>
#include <ArduCAM.h>
#include "model_data.h"       // Your TFLite model as a C array

//#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Undefine any conflicting swap macro from other libraries
#ifdef swap
#undef swap
#endif

// Image dimensions expected by your model.
const int kImageWidth = 96;
const int kImageHeight = 96;
const int kImageChannels = 1;  // Grayscale

// Memory arena for TensorFlow Lite Micro (adjust based on your model size)
const int tensor_arena_size = 10 * 1024;
uint8_t tensor_arena[tensor_arena_size];

// Pointer to the TFLite Micro interpreter and input tensor.
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// ----- Camera Definitions -----
// Adjust the chip select pin as used on your board.
#define CAM_CS_PIN 10
ArduCAM myCAM(OV7675, CAM_CS_PIN);

// Function to initialize the camera.
void initCamera() {
  pinMode(CAM_CS_PIN, OUTPUT);
  digitalWrite(CAM_CS_PIN, HIGH);
  
  SPI.begin();
  
  // Simple SPI test: write and read back a test register.
  myCAM.write_reg(ARDUCHIP_TEST1, 0x55);
  if (myCAM.read_reg(ARDUCHIP_TEST1) != 0x55) {
    Serial.println("SPI interface error! Check wiring and camera module.");
    while (1);
  }
  
  // Additional OV7675 configuration may be required.
  // For example, you might need to call myCAM.InitCAM() or similar.
  Serial.println("Camera initialized.");
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { }  // Wait for serial connection
  
  // Initialize the camera.
  initCamera();

  // Load the TFLite model from model_data.
  const tflite::Model* model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1);
  }

  // Set up the TFLite Micro interpreter.
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, tensor_arena_size);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1);
  }

  // Get pointer to the input tensor.
  input = interpreter->input(0);

  Serial.println("Setup complete. Starting inference loop.");
}

void loop() {
  uint8_t image_data[kImageWidth * kImageHeight];

  // --- Capture image from OV7675 using ArduCAM ---
  // The following functions are common in many ArduCAM examples.
  // If your version uses different names, check the ArduCAM examples.
  myCAM.clear_fifo_flag();       // Clear FIFO flag.
  myCAM.start_capture();         // Begin capture.

  // Wait until capture is complete.
  while (!(myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK))) {
    // Optionally, implement a timeout.
  }
  
  uint32_t length = myCAM.read_fifo_length();
  if (length < (kImageWidth * kImageHeight)) {
    Serial.println("Captured image size is smaller than expected!");
    delay(2000);
    return;
  }
  
  // Read image data from FIFO.
  myCAM.CS_LOW();
  for (int i = 0; i < kImageWidth * kImageHeight; i++) {
    image_data[i] = SPI.transfer(0x00);
  }
  myCAM.CS_HIGH();

  // --- Preprocess and run inference ---
  // Normalize the image data and load it into the input tensor.
  float* input_buffer = input->data.f;
  for (int i = 0; i < kImageWidth * kImageHeight; i++) {
    input_buffer[i] = image_data[i] / 255.0f;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed!");
    delay(2000);
    return;
  }

  // Process the output.
  TfLiteTensor* output = interpreter->output_tensor(0);
  int predicted_class = -1;
  float max_confidence = 0.0;
  for (int i = 0; i < output->dims->data[1]; i++) {
    float confidence = output->data.f[i];
    if (confidence > max_confidence) {
      max_confidence = confidence;
      predicted_class = i;
    }
  }

  Serial.print("Predicted Class: ");
  Serial.print(predicted_class);
  Serial.print(" with confidence: ");
  Serial.print(max_confidence * 100, 1);
  Serial.println("%");

  delay(2000);
}
