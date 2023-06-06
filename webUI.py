import random
import subprocess
import os
import gradio
import gradio as gr
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))


def convert(segment_length, video, audio, progress=gradio.Progress()):
    if segment_length is None:
        segment_length=0
    print(video, audio)

    if segment_length != 0:
        video_segments = cut_video_segments(video, segment_length)
        audio_segments = cut_audio_segments(audio, segment_length)
    else:
        video_path = os.path.join('temp/video', os.path.basename(video))
        shutil.move(video, video_path)
        video_segments = [video_path]
        audio_path = os.path.join('temp/audio', os.path.basename(audio))
        shutil.move(audio, audio_path)
        audio_segments = [audio_path]

    processed_segments = []
    for i, (video_seg, audio_seg) in progress.tqdm(enumerate(zip(video_segments, audio_segments))):
        processed_output = process_segment(video_seg, audio_seg, i)
        processed_segments.append(processed_output)

    output_file = f"results/output_{random.randint(0,1000)}.mp4"
    concatenate_videos(processed_segments, output_file)

    # Remove temporary files
    cleanup_temp_files(video_segments + audio_segments)

    # Return the concatenated video file
    return output_file


def cleanup_temp_files(file_list):
    for file_path in file_list:
        if os.path.isfile(file_path):
            os.remove(file_path)


def cut_video_segments(video_file, segment_length):
    temp_directory = 'temp/audio'
    shutil.rmtree(temp_directory, ignore_errors=True)
    shutil.os.makedirs(temp_directory, exist_ok=True)
    segment_template = f"{temp_directory}/{random.randint(0,1000)}_%03d.mp4"
    command = ["ffmpeg", "-i", video_file, "-c", "copy", "-f",
               "segment", "-segment_time", str(segment_length), segment_template]
    subprocess.run(command, check=True)

    video_segments = [segment_template %
                      i for i in range(len(os.listdir(temp_directory)))]
    return video_segments


def cut_audio_segments(audio_file, segment_length):
    temp_directory = 'temp/video'
    shutil.rmtree(temp_directory, ignore_errors=True)
    shutil.os.makedirs(temp_directory, exist_ok=True)
    segment_template = f"{temp_directory}/{random.randint(0,1000)}_%03d.mp3"
    command = ["ffmpeg", "-i", audio_file, "-f", "segment",
               "-segment_time", str(segment_length), segment_template]
    subprocess.run(command, check=True)

    audio_segments = [segment_template %
                      i for i in range(len(os.listdir(temp_directory)))]
    return audio_segments


def process_segment(video_seg, audio_seg, i):
    output_file = f"results/{random.randint(10,100000)}_{i}.mp4"
    command = ["python", "inference.py", "--face", video_seg,
               "--audio", audio_seg, "--outfile", output_file]
    subprocess.run(command, check=True)

    return output_file


def concatenate_videos(video_segments, output_file):
    with open("segments.txt", "w") as file:
        for segment in video_segments:
            file.write(f"file '{segment}'\n")
    command = ["ffmpeg", "-f", "concat", "-i",
               "segments.txt", "-c", "copy", output_file]
    subprocess.run(command, check=True)


with gradio.Blocks(
    title="Audio-based Lip Synchronization",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        font=["Source Sans Pro", "Arial", "sans-serif"],
        font_mono=['JetBrains mono', "Consolas", 'Courier New']
    ),
) as demo:
    with gradio.Row():
        gradio.Markdown("# Audio-based Lip Synchronization")
    with gradio.Row():
        with gradio.Column():
            with gradio.Row():
                seg = gradio.Number(
                    label="segment length (Second), 0 for no segmentation")
            with gradio.Row():
                with gradio.Column():
                    v = gradio.Video(label='SOurce Face')

                with gradio.Column():
                    a = gradio.Audio(
                        type='filepath', label='Target Audio')

            with gradio.Row():
                btn = gradio.Button(value="Synthesize",variant="primary")
            with gradio.Row():
                gradio.Examples(
                    label="Face Examples",
                    examples=[
                        os.path.join(os.path.dirname(__file__),
                                     "examples/face/1.mp4"),
                        os.path.join(os.path.dirname(__file__),
                                     "examples/face/2.mp4"),
                        os.path.join(os.path.dirname(__file__),
                                     "examples/face/3.mp4"),
                        os.path.join(os.path.dirname(__file__),
                                     "examples/face/4.mp4"),
                        os.path.join(os.path.dirname(__file__),
                                     "examples/face/5.mp4"),
                    ],
                    inputs=[v],
                    fn=convert,
                )
            with gradio.Row():
                gradio.Examples(
                    label="Audio Examples",
                    examples=[
                        os.path.join(os.path.dirname(__file__),
                                     "examples/audio/1.wav"),
                        os.path.join(os.path.dirname(__file__),
                                     "examples/audio/2.wav"),
                    ],
                    inputs=[a],
                    fn=convert,
                )

        with gradio.Column():
            o = gradio.Video(label="Output Video")

    btn.click(fn=convert, inputs=[seg, v, a], outputs=[o])

demo.queue().launch()
