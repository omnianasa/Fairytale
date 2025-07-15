# Fairytale - Neural Style Transfer on Video Frames 

This project explores the application of Neural Style Transfer (NST) to videos by applying the technique frame by frame. The goal is to transform a video into a moving painting by preserving the original content of the video while transferring the artistic style of a famous painting.

---

## Paintings


Below are four of the famous paintings used as style references during experimentation:

| Style Image 1 | Style Image 2 | Style Image 3 | Style Image 4 |
|---------------|---------------|---------------|---------------|
| ![Style 1](samplesForTest/MemPersistence.jpg) | ![Style 2](samplesForTest/rainRustle.jpg) | ![Style 3](samplesForTest/Starry.jpg) | ![Style 4](samplesForTest/waterLILI.jpg) |

---

## Samples output 

![original image](samplesForTest/frame_0013.jpg)
<br>

after applying the 4 styles on we have new amazing painted images shown below

| Image 1 |  Image 2 |  Image 3 |  Image 4 |
|---------------|---------------|---------------|---------------|
| ![Image 1](samplesForTest/image_MemPersistence.jpg) | ![Image 2](samplesForTest/image_rain_rustle.jpg) | ![Image 3](samplesForTest/image_starry.jpg) | ![Image 4](samplesForTest/image_waterLILI.jpg) |



## Stylized Output Frames

Each of the 26 frames was stylized using the selected style image. Below is a preview grid of the stylized frames(some of them):

| Frame 0 | Frame 1 | Frame 2 | Frame 3 |
|--------|--------|--------|--------|
| ![F0](stylized_frames/stylized_frame_0000.jpg) | ![F1](stylized_frames/stylized_frame_0001.jpg) | ![F2](stylized_frames/stylized_frame_0002.jpg) | ![F3](stylized_frames/stylized_frame_0003.jpg) |

| Frame 4 | Frame 5 | Frame 6 | Frame 7 |
|--------|--------|--------|--------|
| ![F4](stylized_frames/stylized_frame_0004.jpg) | ![F5](stylized_frames/stylized_frame_0005.jpg) | ![F6](stylized_frames/stylized_frame_0006.jpg) | ![F7](stylized_frames/stylized_frame_0007.jpg) |


---

## Final output

After processing the frames, the stylized images were combined back into a video:

**Output video preview:**  
![Final Video](stylized_output.mp4)  

![Watch on youtube](https://youtube.com/shorts/A7FXdvtBR6k?feature=share)

---

