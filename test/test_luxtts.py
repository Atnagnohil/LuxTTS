"""
LuxTTS 本地测试脚本
用法: python test_luxtts.py --audio <参考音频路径> --text <要合成的文字>
"""
import argparse
import sys

MODEL_PATH = "/home/host/.cache/huggingface/hub/models--YatharthS--LuxTTS/snapshots/527f245a276a0eb42ea103a7a512bcfd771eb9b6"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="参考音频路径 (wav/mp3, 至少3秒)")
    parser.add_argument("--text", default="Hello, this is a test of LuxTTS voice cloning.", help="要合成的文字")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--output", default="output.wav", help="输出文件路径")
    parser.add_argument("--steps", type=int, default=4, help="推理步数 (3-4 推荐)")
    parser.add_argument("--t_shift", type=float, default=0.9, help="采样参数 (0.5-1.0)")
    args = parser.parse_args()

    print(f"加载模型: {MODEL_PATH}")
    print(f"设备: {args.device}")

    from zipvoice.luxvoice import LuxTTS
    lux_tts = LuxTTS(MODEL_PATH, device=args.device)

    print(f"编码参考音频: {args.audio}")
    encoded_prompt = lux_tts.encode_prompt(args.audio, rms=0.01)

    print(f"生成语音: {args.text}")
    wav = lux_tts.generate_speech(
        args.text,
        encoded_prompt,
        num_steps=args.steps,
        t_shift=args.t_shift,
    )

    import soundfile as sf
    wav_np = wav.numpy().squeeze()
    sf.write(args.output, wav_np, 48000)
    print(f"✓ 输出已保存: {args.output}  (时长: {len(wav_np)/48000:.2f}s)")


if __name__ == "__main__":
    main()
