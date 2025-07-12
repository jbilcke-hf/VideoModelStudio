# Video Model Training Notes

## Training Step Analysis

### What happens in a training step?

A training step processes **exactly `batch_size` samples** (not the entire dataset). Here's what happens:

**Per Training Step:**
- Processes `batch_size` videos/samples (configurable, typically 1-8)
- Uses smart batching that groups videos by resolution dimensions
- Two data streams: text embeddings + video latents

**Key Points:**
- With 100 videos and batch_size=4: each step processes 4 videos
- Training runs for a fixed number of steps (not epochs)
- Dataset loops infinitely, so videos are reused across steps
- Uses ResolutionSampler to batch videos of similar dimensions together

**Training Loop Structure:**
1. Load next `batch_size` samples from dataset
2. Group by resolution (spatial + temporal dimensions)  
3. Forward pass through transformer (denoising)
4. Calculate loss and update weights
5. Increment step counter

So if you have 100 videos and batch_size=1, step 1 processes video 1, step 2 processes video 2, etc. When it reaches video 100, it loops back to video 1.

## Avoiding Overfitting

For video model training, a good rule of thumb is to keep each video seen **less than 10-50 times** during training to avoid overfitting.

**Common thresholds:**
- **Conservative**: <10 times per video (strong generalization)
- **Moderate**: 10-50 times per video (balanced)
- **Risky**: >100 times per video (likely overfitting)

**With low learning rates (e.g., 0.00004):**
- Lower LR means you can potentially see videos more times safely
- But still better to err on the side of caution

**Practical calculation:**
- If training for 10,000 steps with batch_size=1:
  - 100 videos = 100 times each (risky)
  - 500 videos = 20 times each (moderate)
  - 1,000+ videos = <10 times each (conservative)

**Early stopping indicators:**
- Training loss continues decreasing but validation loss plateaus/increases
- Generated videos start looking too similar to training examples
- Loss of diversity in outputs

With low learning rates, staying under 20-30 times per video should be relatively safe, but <10 times is ideal for strong generalization.