// Reverse Image Filtering Implementation - Direct reverse filtering without initial filtering
// FFT.js library required: https://cdnjs.cloudflare.com/ajax/libs/fft.js/4.0.3/fft.min.js

document.addEventListener("DOMContentLoaded", function () {
  const originalImageContainer = document.getElementById("originalImage");
  const filteredImageContainer = document.getElementById("filteredImage");
  const restoredImageContainer = document.getElementById("restoredImage");
  const processButton = document.getElementById("processButton");
  const loadingIndicator = document.getElementById("loadingIndicator");
  const similarityContainer = document.getElementById("similarityContainer");
  const similarityScore = document.getElementById("similarityScore");
  const similarityBar = document.getElementById("similarityBar");

  // Get the uploaded image from localStorage if available
  const uploadedImageData = localStorage.getItem("uploadedImage");
  if (uploadedImageData) {
    filteredImageContainer.innerHTML = `<img src="${uploadedImageData}" alt="Input Image">`;
  }

  processButton.addEventListener("click", function () {
    // Show loading indicator
    loadingIndicator.classList.remove("hidden");
    processButton.disabled = true;

    // Get the image from the filtered image container
    const inputImage = filteredImageContainer.querySelector("img");

    if (inputImage) {
      // Create an offscreen canvas to work with the image
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");

      // Create a new image to properly load the input image
      const img = new Image();
      img.crossOrigin = "Anonymous";
      img.src = inputImage.src;

      img.onload = function () {
        // Set canvas dimensions to match the image
        canvas.width = img.width;
        canvas.height = img.height;

        // Draw the image on the canvas
        ctx.drawImage(img, 0, 0);

        // Get image data for processing
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        // Process the image using direct reverse filtering
        const restoredImageData = reverseImageFilter(imageData);

        // Create another canvas for the restored image
        const restoredCanvas = document.createElement("canvas");
        restoredCanvas.width = canvas.width;
        restoredCanvas.height = canvas.height;
        const restoredCtx = restoredCanvas.getContext("2d");

        // Put the processed data back into the canvas
        restoredCtx.putImageData(restoredImageData, 0, 0);

        // Display both the original input image and the restored image
        originalImageContainer.innerHTML = `<img src="${inputImage.src}" alt="Original Input Image">`;
        restoredImageContainer.innerHTML = `<img src="${restoredCanvas.toDataURL()}" alt="Restored Image">`;

        // Calculate similarity between original and restored image
        const similarityPercentage = calculateImageSimilarity(
          imageData,
          restoredImageData
        );

        // Update similarity display
        similarityScore.textContent = `${similarityPercentage.toFixed(1)}%`;
        similarityBar.style.width = `${similarityPercentage}%`;

        // Set color based on similarity percentage
        if (similarityPercentage >= 80) {
          similarityBar.className = "similarity-fill high";
        } else if (similarityPercentage >= 60) {
          similarityBar.className = "similarity-fill medium";
        } else {
          similarityBar.className = "similarity-fill low";
        }

        // Show results container and similarity info
        document.getElementById("resultsContainer").classList.remove("hidden");
        similarityContainer.classList.remove("hidden");

        // Hide loading indicator
        loadingIndicator.classList.add("hidden");
        processButton.disabled = false;
      };
    }
  });

  // Function to calculate similarity between two images
  function calculateImageSimilarity(imageData1, imageData2) {
    const pixels1 = imageData1.data;
    const pixels2 = imageData2.data;
    const length = pixels1.length;

    let totalDifference = 0;
    let maxPossibleDifference = 0;

    // Compare each pixel channel (RGB only, skip alpha)
    for (let i = 0; i < length; i += 4) {
      // Calculate difference for RGB channels
      for (let c = 0; c < 3; c++) {
        const channelDiff = Math.abs(pixels1[i + c] - pixels2[i + c]);
        totalDifference += channelDiff;
        maxPossibleDifference += 255; // Max possible difference per channel
      }
    }

    // Calculate similarity as percentage (inverted difference)
    const similarityPercentage =
      100 - (totalDifference / maxPossibleDifference) * 100;

    return similarityPercentage;
  }

  // Algorithm: Direct reverse image filtering without initial Gaussian filtering
  function reverseImageFilter(imageData) {
    const width = imageData.width;
    const height = imageData.height;
    const pixels = imageData.data;

    console.log("Starting: Direct reverse image filtering...");

    try {
      // Input: Original image (no pre-filtering applied)
      const N = 20; // Number of iterations

      // Extract RGB channels and convert to normalized values [0,1]
      const totalPixels = width * height;
      const Y_R = new Float64Array(totalPixels);
      const Y_G = new Float64Array(totalPixels);
      const Y_B = new Float64Array(totalPixels);

      for (let i = 0; i < pixels.length; i += 4) {
        const idx = Math.floor(i / 4);
        if (idx < totalPixels) {
          Y_R[idx] = (pixels[i] || 0) / 255; // R channel
          Y_G[idx] = (pixels[i + 1] || 0) / 255; // G channel
          Y_B[idx] = (pixels[i + 2] || 0) / 255; // B channel
        }
      }

      // Define the assumed filter characteristics for reverse filtering
      const filterSize = 7;
      const sigma = 2.0;
      const gaussianKernel = createGaussianKernel(filterSize, sigma);

      // Step 1: X̂^(0) ← Y (Initialize with input image)
      let X_hat_R = new Float64Array(Y_R);
      let X_hat_G = new Float64Array(Y_G);
      let X_hat_B = new Float64Array(Y_B);

      // Step 2-4: Iterative reverse filtering process
      for (let t = 0; t < N; t++) {
        console.log(`Iteration ${t + 1}/${N}`);

        // Apply the assumed filter to current estimate X̂^(t)
        const f_X_hat_R = applyFilter(
          X_hat_R,
          width,
          height,
          gaussianKernel,
          filterSize
        );
        const f_X_hat_G = applyFilter(
          X_hat_G,
          width,
          height,
          gaussianKernel,
          filterSize
        );
        const f_X_hat_B = applyFilter(
          X_hat_B,
          width,
          height,
          gaussianKernel,
          filterSize
        );

        // Step 3: Update estimate using reverse filtering formula
        X_hat_R = updateEstimate(Y_R, X_hat_R, f_X_hat_R, width, height);
        X_hat_G = updateEstimate(Y_G, X_hat_G, f_X_hat_G, width, height);
        X_hat_B = updateEstimate(Y_B, X_hat_B, f_X_hat_B, width, height);
      }

      // Step 5: Return final estimate
      // Convert back to ImageData format
      const restoredImageData = new ImageData(width, height);
      const restoredPixels = restoredImageData.data;

      for (let i = 0; i < totalPixels; i++) {
        const idx = i * 4;
        // Clamp values to [0,1] and convert back to [0,255] range
        restoredPixels[idx] = Math.max(0, Math.min(255, X_hat_R[i] * 255)); // R
        restoredPixels[idx + 1] = Math.max(0, Math.min(255, X_hat_G[i] * 255)); // G
        restoredPixels[idx + 2] = Math.max(0, Math.min(255, X_hat_B[i] * 255)); // B
        restoredPixels[idx + 3] = 255; // Alpha channel
      }

      console.log("Direct reverse filtering completed.");
      return restoredImageData;
    } catch (error) {
      console.error("Error:", error);
      // Fallback to simple sharpening if the algorithm fails
      return applyBasicSharpening(imageData);
    }
  }

  // Update estimate using reverse filtering formula
  function updateEstimate(Y, X_hat, f_X_hat, width, height) {
    try {
      // Compute FFT of Y, X̂^(t), and f(X̂^(t))
      const F_Y = fft2D(Y, width, height);
      const F_X_hat = fft2D(X_hat, width, height);
      const F_f_X_hat = fft2D(f_X_hat, width, height);

      // Element-wise operations: (F(Y) · F(X̂^(t))) / F(f(X̂^(t)))
      const result_fft = new Array(width * height);
      const eps = 1e-10; // Small epsilon to avoid division by zero

      for (let i = 0; i < width * height; i++) {
        // Complex multiplication: F(Y) * F(X̂^(t))
        const numeratorReal =
          F_Y[i].real * F_X_hat[i].real - F_Y[i].imag * F_X_hat[i].imag;
        const numeratorImag =
          F_Y[i].real * F_X_hat[i].imag + F_Y[i].imag * F_X_hat[i].real;

        // Complex division: numerator / F(f(X̂^(t)))
        const denominatorReal = F_f_X_hat[i].real + eps;
        const denominatorImag = F_f_X_hat[i].imag;
        const denominatorMagnitude =
          denominatorReal * denominatorReal + denominatorImag * denominatorImag;

        result_fft[i] = {
          real:
            (numeratorReal * denominatorReal +
              numeratorImag * denominatorImag) /
            denominatorMagnitude,
          imag:
            (numeratorImag * denominatorReal -
              numeratorReal * denominatorImag) /
            denominatorMagnitude,
        };
      }

      // Inverse FFT to get the updated estimate
      return ifft2D(result_fft, width, height);
    } catch (error) {
      console.error("Error in updateEstimate:", error);
      return X_hat; // Return unchanged if error occurs
    }
  }

  // Apply filter - convolution with kernel (used for reverse filtering estimation)
  function applyFilter(channel, width, height, kernel, kernelSize) {
    const halfKernelSize = Math.floor(kernelSize / 2);
    const result = new Float64Array(channel.length);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = 0;

        for (let ky = 0; ky < kernelSize; ky++) {
          for (let kx = 0; kx < kernelSize; kx++) {
            // Calculate pixel coordinates with padding
            let pixelX = x + kx - halfKernelSize;
            let pixelY = y + ky - halfKernelSize;

            // Use circular padding for convolution
            pixelX = ((pixelX % width) + width) % width;
            pixelY = ((pixelY % height) + height) % height;

            const kernelValue = kernel[ky * kernelSize + kx];
            const pixelValue = channel[pixelY * width + pixelX];

            sum += pixelValue * kernelValue;
          }
        }

        result[y * width + x] = sum;
      }
    }

    return result;
  }

  // Create Gaussian kernel for reverse filtering estimation
  function createGaussianKernel(size, sigma) {
    const kernel = new Float64Array(size * size);
    const center = Math.floor(size / 2);
    let sum = 0;

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const dx = x - center;
        const dy = y - center;
        const value = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
        kernel[y * size + x] = value;
        sum += value;
      }
    }

    // Normalize the kernel so sum equals 1
    for (let i = 0; i < kernel.length; i++) {
      kernel[i] /= sum;
    }

    return kernel;
  }

  // 2D FFT implementation using Cooley-Tukey algorithm
  function fft2D(data, width, height) {
    // Convert input data to complex format
    const complexData = new Array(width * height);
    for (let i = 0; i < width * height; i++) {
      complexData[i] = { real: data[i], imag: 0 };
    }

    // Apply row-wise FFT
    for (let row = 0; row < height; row++) {
      const rowData = complexData.slice(row * width, (row + 1) * width);
      const rowFFT = fft1D(rowData);
      for (let col = 0; col < width; col++) {
        complexData[row * width + col] = rowFFT[col];
      }
    }

    // Apply column-wise FFT
    for (let col = 0; col < width; col++) {
      const colData = new Array(height);
      for (let row = 0; row < height; row++) {
        colData[row] = complexData[row * width + col];
      }
      const colFFT = fft1D(colData);
      for (let row = 0; row < height; row++) {
        complexData[row * width + col] = colFFT[row];
      }
    }

    return complexData;
  }

  // 2D Inverse FFT implementation
  function ifft2D(complexData, width, height) {
    // Conjugate the input
    const conjugated = complexData.map((x) => ({
      real: x.real,
      imag: -x.imag,
    }));

    // Apply 2D FFT to conjugated data
    const fftResult = fft2D_complex(conjugated, width, height);

    // Conjugate and normalize the result, return only real parts
    const result = new Float64Array(width * height);
    const normFactor = 1 / (width * height);

    for (let i = 0; i < width * height; i++) {
      result[i] = fftResult[i].real * normFactor; // Take real part and normalize
    }

    return result;
  }

  // 2D FFT for complex input data
  function fft2D_complex(complexData, width, height) {
    const result = [...complexData];

    // Row-wise FFT
    for (let row = 0; row < height; row++) {
      const rowData = result.slice(row * width, (row + 1) * width);
      const rowFFT = fft1D(rowData);
      for (let col = 0; col < width; col++) {
        result[row * width + col] = rowFFT[col];
      }
    }

    // Column-wise FFT
    for (let col = 0; col < width; col++) {
      const colData = new Array(height);
      for (let row = 0; row < height; row++) {
        colData[row] = result[row * width + col];
      }
      const colFFT = fft1D(colData);
      for (let row = 0; row < height; row++) {
        result[row * width + col] = colFFT[row];
      }
    }

    return result;
  }

  // 1D FFT implementation using Cooley-Tukey algorithm
  function fft1D(data) {
    const N = data.length;
    if (N <= 1) return data;

    // Pad to next power of 2 if necessary
    const nextPow2 = Math.pow(2, Math.ceil(Math.log2(N)));
    if (N < nextPow2) {
      const padded = [...data];
      for (let i = N; i < nextPow2; i++) {
        padded.push({ real: 0, imag: 0 });
      }
      return fft1D(padded).slice(0, N);
    }

    // Divide and conquer
    const even = new Array(N / 2);
    const odd = new Array(N / 2);

    for (let i = 0; i < N / 2; i++) {
      even[i] = data[2 * i];
      odd[i] = data[2 * i + 1];
    }

    const evenFFT = fft1D(even);
    const oddFFT = fft1D(odd);

    const result = new Array(N);

    for (let k = 0; k < N / 2; k++) {
      // Twiddle factor: e^(-2πik/N)
      const thetaReal = Math.cos((-2 * Math.PI * k) / N);
      const thetaImag = Math.sin((-2 * Math.PI * k) / N);

      // Complex multiplication: theta * oddFFT[k]
      const t = {
        real: thetaReal * oddFFT[k].real - thetaImag * oddFFT[k].imag,
        imag: thetaReal * oddFFT[k].imag + thetaImag * oddFFT[k].real,
      };

      result[k] = {
        real: evenFFT[k].real + t.real,
        imag: evenFFT[k].imag + t.imag,
      };

      result[k + N / 2] = {
        real: evenFFT[k].real - t.real,
        imag: evenFFT[k].imag - t.imag,
      };
    }

    return result;
  }

  // Fallback function for basic image sharpening
  function applyBasicSharpening(imageData) {
    const width = imageData.width;
    const height = imageData.height;
    const pixels = new Uint8ClampedArray(imageData.data);
    const result = new ImageData(width, height);
    const resultPixels = result.data;

    // Simple sharpening kernel
    const kernel = [0, -1, 0, -1, 5, -1, 0, -1, 0];

    // Apply convolution for sharpening
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        for (let c = 0; c < 3; c++) {
          // For R, G, B channels
          let sum = 0;

          for (let ky = -1; ky <= 1; ky++) {
            for (let kx = -1; kx <= 1; kx++) {
              const idx = ((y + ky) * width + (x + kx)) * 4 + c;
              sum += pixels[idx] * kernel[(ky + 1) * 3 + (kx + 1)];
            }
          }

          // Set the result
          resultPixels[(y * width + x) * 4 + c] = Math.max(
            0,
            Math.min(255, sum)
          );
        }

        // Set alpha channel
        resultPixels[(y * width + x) * 4 + 3] = 255;
      }
    }

    // Copy edge pixels from original
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (x === 0 || x === width - 1 || y === 0 || y === height - 1) {
          const idx = (y * width + x) * 4;
          resultPixels[idx] = pixels[idx];
          resultPixels[idx + 1] = pixels[idx + 1];
          resultPixels[idx + 2] = pixels[idx + 2];
          resultPixels[idx + 3] = pixels[idx + 3];
        }
      }
    }

    return result;
  }
});
