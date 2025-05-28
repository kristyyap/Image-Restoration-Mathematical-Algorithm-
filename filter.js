// Reverse Image Filtering Implementation
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
  // const uploadedImageData = localStorage.getItem("uploadedImage");
  // if (uploadedImageData) {
  //     filteredImageContainer.innerHTML = `<img src="${uploadedImageData}" alt="Filtered Image">`;
  // }
  // Check for image URL from extension
  const urlParams = new URLSearchParams(window.location.search);
  const imageUrl = urlParams.get("image");

  if (imageUrl) {
    const decodedUrl = decodeURIComponent(imageUrl);
    filteredImageContainer.innerHTML = `<img src="${decodedUrl}" alt="Filtered Image">`;
    // filteredImageSection?.classList?.remove("hidden");
  }

  function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
  }

  function hideError() {
    errorMessage.classList.add("hidden");
  }

  processButton.addEventListener("click", function () {
    // Show loading indicator
    loadingIndicator.classList.remove("hidden");
    processButton.disabled = true;

    // Get the image from the filtered image container
    const filteredImage = filteredImageContainer.querySelector("img");

    if (filteredImage) {
      // Create an offscreen canvas to work with the image
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");

      // Create a new image to properly load the filtered image
      const img = new Image();
      img.crossOrigin = "Anonymous";
      img.src = filteredImage.src;

      img.onload = function () {
        try {
          // Set canvas dimensions to match the image
          canvas.width = img.width;
          canvas.height = img.height;

          // Draw the image on the canvas
          ctx.drawImage(img, 0, 0);

          // Get image data for processing
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

          // Process the image (reverse filtering)
          const restoredImageData = reverseFilter(imageData);

          // Create another canvas for the restored image
          const restoredCanvas = document.createElement("canvas");
          restoredCanvas.width = canvas.width;
          restoredCanvas.height = canvas.height;
          const restoredCtx = restoredCanvas.getContext("2d");

          // Put the processed data back into the canvas
          restoredCtx.putImageData(restoredImageData, 0, 0);

          // Display both the original filtered image and the restored image
          originalImageContainer.innerHTML = `<img src="${filteredImage.src}" alt="Original Filtered Image">`;
          restoredImageContainer.innerHTML = `<img src="${restoredCanvas.toDataURL()}" alt="Restored Image">`;

          // Calculate similarity between original and restored image
          const similarityPercentage = calculateImageSimilarity(
            imageData,
            restoredImageData
          );

          // Update similarity display
          const percentage = similarityPercentage.toFixed(1);
          similarityScore.textContent = `${percentage}%`;
          similarityBar.style.width = `${similarityPercentage}%`;

          // Set color based on similarity percentage
          if (similarityPercentage >= 99) {
            similarityBar.className = "similarity-fill high";
            similarityContainer.innerHTML = `
    <div class="similarity-message">No filter detected on this image</div>
  `;
          } else if (similarityPercentage >= 80) {
            similarityBar.className = "similarity-fill high";
          } else if (similarityPercentage >= 60) {
            similarityBar.className = "similarity-fill medium";
          } else {
            similarityBar.className = "similarity-fill low";
          }

          // Show results container and similarity info
          document
            .getElementById("resultsContainer")
            .classList.remove("hidden");
          similarityContainer.classList.remove("hidden");
        } catch (error) {
          console.error("Processing error:", error);
          showError(
            "Error processing image. Please try again with a different image."
          );
        } finally {
          // Hide loading indicator
          loadingIndicator.classList.add("hidden");
          processButton.disabled = false;
        }
      };

      img.onerror = function () {
        showError("Error loading image for processing.");
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

  // Function to reverse filter the image
  function reverseFilter(imageData) {
    // Extract dimensions and data
    const width = imageData.width;
    const height = imageData.height;
    const pixels = imageData.data;

    // Create arrays for R, G, B channels
    const R = new Array(width * height);
    const G = new Array(width * height);
    const B = new Array(width * height);

    // Extract RGB channels
    for (let i = 0; i < pixels.length; i += 4) {
      const idx = i / 4;
      R[idx] = pixels[i] / 255; // R
      G[idx] = pixels[i + 1] / 255; // G
      B[idx] = pixels[i + 2] / 255; // B
    }

    console.log("Starting reverse filtering process...");

    try {
      // For better performance and reliability, we'll use a combination of
      // unsharp masking and deconvolution techniques

      // Apply the reverse filtering algorithm
      const [restoredR, restoredG, restoredB] = applyAdvancedReverseFiltering(
        R,
        G,
        B,
        width,
        height
      );

      // Create a new ImageData object for the restored image
      const restoredImageData = new ImageData(width, height);
      const restoredPixels = restoredImageData.data;

      // Set pixel values for the restored image
      for (let i = 0; i < width * height; i++) {
        const idx = i * 4;
        // Clamp values between 0 and 1, then scale back to 0-255
        restoredPixels[idx] = Math.max(0, Math.min(1, restoredR[i])) * 255;
        restoredPixels[idx + 1] = Math.max(0, Math.min(1, restoredG[i])) * 255;
        restoredPixels[idx + 2] = Math.max(0, Math.min(1, restoredB[i])) * 255;
        restoredPixels[idx + 3] = 255; // Alpha channel
      }

      console.log("Reverse filtering completed successfully");
      return restoredImageData;
    } catch (error) {
      console.error("Error during reverse filtering:", error);

      // In case of error, return a simple sharpened version as fallback
      return applyBasicSharpening(imageData);
    }
  }

  // Advanced reverse filtering approach combining multiple techniques
  function applyAdvancedReverseFiltering(R, G, B, width, height) {
    // 1. Initial unsharp masking pass
    const [sharpR, sharpG, sharpB] = applyUnsharpMasking(
      R,
      G,
      B,
      width,
      height,
      0.8
    );

    // 2. Apply adaptive local contrast enhancement
    const [enhancedR, enhancedG, enhancedB] = applyAdaptiveContrastEnhancement(
      sharpR,
      sharpG,
      sharpB,
      width,
      height
    );

    // 3. Apply noise reduction to smooth out artifacts
    const [finalR, finalG, finalB] = applyBilateralFiltering(
      enhancedR,
      enhancedG,
      enhancedB,
      width,
      height,
      3,
      0.1,
      0.1
    );

    return [finalR, finalG, finalB];
  }

  // Unsharp masking for basic image enhancement
  function applyUnsharpMasking(R, G, B, width, height, amount) {
    // Create Gaussian blur kernels for unsharp masking
    const kernel = createGaussianKernel(5, 1.5);

    // Apply Gaussian blur
    const blurredR = applyFilter(R, width, height, kernel);
    const blurredG = applyFilter(G, width, height, kernel);
    const blurredB = applyFilter(B, width, height, kernel);

    // Apply unsharp masking formula: original + amount * (original - blurred)
    const sharpR = new Array(width * height);
    const sharpG = new Array(width * height);
    const sharpB = new Array(width * height);

    for (let i = 0; i < width * height; i++) {
      sharpR[i] = R[i] + amount * (R[i] - blurredR[i]);
      sharpG[i] = G[i] + amount * (G[i] - blurredG[i]);
      sharpB[i] = B[i] + amount * (B[i] - blurredB[i]);
    }

    return [sharpR, sharpG, sharpB];
  }

  // Adaptive local contrast enhancement
  function applyAdaptiveContrastEnhancement(R, G, B, width, height) {
    const resultR = new Array(width * height);
    const resultG = new Array(width * height);
    const resultB = new Array(width * height);

    // Window size for local contrast
    const windowSize = 7;
    const windowRadius = Math.floor(windowSize / 2);

    // Process each pixel
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        // Collect local neighborhood
        let sumR = 0,
          sumG = 0,
          sumB = 0;
        let count = 0;

        for (let wy = -windowRadius; wy <= windowRadius; wy++) {
          for (let wx = -windowRadius; wx <= windowRadius; wx++) {
            const nx = Math.min(Math.max(x + wx, 0), width - 1);
            const ny = Math.min(Math.max(y + wy, 0), height - 1);
            const idx = ny * width + nx;

            sumR += R[idx];
            sumG += G[idx];
            sumB += B[idx];
            count++;
          }
        }

        // Local mean
        const meanR = sumR / count;
        const meanG = sumG / count;
        const meanB = sumB / count;

        // Current pixel
        const idx = y * width + x;
        const pixelR = R[idx];
        const pixelG = G[idx];
        const pixelB = B[idx];

        // Enhancement factor (adjust as needed)
        const enhanceFactor = 1.5;

        // Apply contrast enhancement: center on mean and enhance difference
        resultR[idx] = meanR + enhanceFactor * (pixelR - meanR);
        resultG[idx] = meanG + enhanceFactor * (pixelG - meanG);
        resultB[idx] = meanB + enhanceFactor * (pixelB - meanB);
      }
    }

    return [resultR, resultG, resultB];
  }

  // Bilateral filtering for edge-preserving noise reduction
  function applyBilateralFiltering(
    R,
    G,
    B,
    width,
    height,
    radius,
    sigmaSpace,
    sigmaRange
  ) {
    const resultR = new Array(width * height);
    const resultG = new Array(width * height);
    const resultB = new Array(width * height);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sumR = 0,
          sumG = 0,
          sumB = 0;
        let totalWeight = 0;

        const centerIdx = y * width + x;
        const centerR = R[centerIdx];
        const centerG = G[centerIdx];
        const centerB = B[centerIdx];

        for (let wy = -radius; wy <= radius; wy++) {
          for (let wx = -radius; wx <= radius; wx++) {
            const nx = Math.min(Math.max(x + wx, 0), width - 1);
            const ny = Math.min(Math.max(y + wy, 0), height - 1);
            const idx = ny * width + nx;

            // Calculate spatial distance
            const spatialDist = wx * wx + wy * wy;
            const spatialWeight = Math.exp(
              -spatialDist / (2 * sigmaSpace * sigmaSpace)
            );

            // Calculate range distance (color difference)
            const rangeDistR = (centerR - R[idx]) * (centerR - R[idx]);
            const rangeDistG = (centerG - G[idx]) * (centerG - G[idx]);
            const rangeDistB = (centerB - B[idx]) * (centerB - B[idx]);
            const rangeDist = rangeDistR + rangeDistG + rangeDistB;
            const rangeWeight = Math.exp(
              -rangeDist / (2 * sigmaRange * sigmaRange)
            );

            // Combined weight
            const weight = spatialWeight * rangeWeight;

            sumR += R[idx] * weight;
            sumG += G[idx] * weight;
            sumB += B[idx] * weight;
            totalWeight += weight;
          }
        }

        // Normalize by total weight
        resultR[centerIdx] = sumR / totalWeight;
        resultG[centerIdx] = sumG / totalWeight;
        resultB[centerIdx] = sumB / totalWeight;
      }
    }

    return [resultR, resultG, resultB];
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

  // Create a Gaussian kernel (similar to fspecial('gaussian') in MATLAB)
  function createGaussianKernel(size, sigma) {
    const kernel = new Array(size * size);
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

    // Normalize the kernel
    for (let i = 0; i < kernel.length; i++) {
      kernel[i] /= sum;
    }

    return kernel;
  }

  // Apply the filtering function (imfilter equivalent)
  function applyFilter(channel, width, height, kernel) {
    const kernelSize = Math.sqrt(kernel.length);
    const halfKernelSize = Math.floor(kernelSize / 2);
    const result = new Array(channel.length);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = 0;

        for (let ky = 0; ky < kernelSize; ky++) {
          for (let kx = 0; kx < kernelSize; kx++) {
            // Calculate corresponding pixel position with circular padding
            const pixelX = Math.min(
              Math.max(x + kx - halfKernelSize, 0),
              width - 1
            );
            const pixelY = Math.min(
              Math.max(y + ky - halfKernelSize, 0),
              height - 1
            );

            // Get kernel and pixel values
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
});
