"""
Gentle-style HTML visualization.

Generates an interactive HTML page for visualizing alignment results,
based on Gentle aligner's visualization:
https://github.com/lowerquality/gentle/blob/master/serve.py

The visualization shows:
- Waveform (if audio is provided)
- Text with highlighted aligned words
- Click on words to play corresponding audio
"""

from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import json
import base64
import os


def get_gentle_visualization(
    word_alignment: Dict[int, Any],
    text: str,
    audio_file: Optional[Union[str, Path]] = None,
    frame_duration: float = 0.02,
    i_word_start: int = 0,
    i_word_end: Optional[int] = None,
    title: str = "TorchAudio Aligner - Alignment Visualization",
) -> str:
    """
    Generate Gentle-style HTML visualization.

    Args:
        word_alignment: Dict mapping word index to AlignedWord
        text: Original text (for display)
        audio_file: Path to audio file (optional)
        frame_duration: Duration of each frame in seconds
        i_word_start: Start word index in text
        i_word_end: End word index in text
        title: Page title

    Returns:
        HTML content string
    """
    text_words = text.split()
    if i_word_end is None:
        i_word_end = len(text_words)

    # Build word data
    words_data = []
    for i, word in enumerate(text_words[i_word_start:i_word_end], start=i_word_start):
        word_info = {
            "word": word,
            "index": i,
            "aligned": False,
            "start": None,
            "end": None,
        }

        if i in word_alignment:
            aligned = word_alignment[i]
            # AlignedWord now has start_seconds()/end_seconds() methods
            start = aligned.start_seconds(frame_duration)
            end = aligned.end_seconds(frame_duration)

            word_info["aligned"] = True
            word_info["start"] = start
            word_info["end"] = end

        words_data.append(word_info)

    # Build HTML
    html = _build_gentle_html(words_data, audio_file, title)
    return html


def _build_gentle_html(
    words_data: List[Dict],
    audio_file: Optional[Union[str, Path]],
    title: str,
) -> str:
    """Build the HTML content."""
    words_json = json.dumps(words_data)

    # Audio source
    if audio_file:
        audio_path = str(audio_file)
        audio_html = f'<audio id="audio" controls src="{audio_path}"></audio>'
    else:
        audio_html = '<p style="color: #888;">No audio file provided</p>'

    # Generate word spans
    word_spans = []
    for w in words_data:
        if w["aligned"]:
            span = f'<span class="word aligned" data-start="{w["start"]}" data-end="{w["end"]}">{w["word"]}</span>'
        else:
            span = f'<span class="word unaligned">{w["word"]}</span>'
        word_spans.append(span)

    words_html = " ".join(word_spans)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .audio-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        audio {{
            width: 100%;
        }}
        .transcript {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            line-height: 2;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .word {{
            display: inline;
            cursor: pointer;
            padding: 2px 4px;
            margin: 2px;
            border-radius: 3px;
            transition: background 0.2s;
        }}
        .word.aligned {{
            background: #e8f5e9;
            color: #2e7d32;
        }}
        .word.aligned:hover {{
            background: #c8e6c9;
        }}
        .word.aligned.playing {{
            background: #4CAF50;
            color: white;
        }}
        .word.unaligned {{
            background: #ffebee;
            color: #c62828;
        }}
        .stats {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats span {{
            margin-right: 20px;
        }}
        .legend {{
            margin-top: 20px;
            font-size: 14px;
            color: #666;
        }}
        .legend span {{
            padding: 3px 8px;
            border-radius: 3px;
            margin-right: 10px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>

    <div class="stats">
        <span><strong>Total words:</strong> {len(words_data)}</span>
        <span><strong>Aligned:</strong> {sum(1 for w in words_data if w['aligned'])}</span>
        <span><strong>Unaligned:</strong> {sum(1 for w in words_data if not w['aligned'])}</span>
    </div>

    <div class="audio-container">
        <h3>Audio Player</h3>
        {audio_html}
    </div>

    <div class="transcript">
        <h3>Transcript</h3>
        <p>{words_html}</p>
    </div>

    <div class="legend">
        <span class="word aligned">Aligned word</span>
        <span class="word unaligned">Unaligned word</span>
        <p style="margin-top: 10px;">Click on aligned words to play the corresponding audio segment.</p>
    </div>

    <script>
        const audio = document.getElementById('audio');
        const words = document.querySelectorAll('.word.aligned');

        words.forEach(word => {{
            word.addEventListener('click', () => {{
                const start = parseFloat(word.dataset.start);
                const end = parseFloat(word.dataset.end);

                if (audio && !isNaN(start)) {{
                    audio.currentTime = start;
                    audio.play();

                    // Highlight current word
                    words.forEach(w => w.classList.remove('playing'));
                    word.classList.add('playing');

                    // Stop at end of word
                    const duration = (end - start) * 1000;
                    setTimeout(() => {{
                        if (audio.currentTime >= end - 0.1) {{
                            audio.pause();
                        }}
                        word.classList.remove('playing');
                    }}, duration + 100);
                }}
            }});
        }});

        // Highlight current word during playback
        if (audio) {{
            audio.addEventListener('timeupdate', () => {{
                const currentTime = audio.currentTime;
                words.forEach(word => {{
                    const start = parseFloat(word.dataset.start);
                    const end = parseFloat(word.dataset.end);
                    if (currentTime >= start && currentTime < end) {{
                        word.classList.add('playing');
                    }} else {{
                        word.classList.remove('playing');
                    }}
                }});
            }});
        }}
    </script>
</body>
</html>"""

    return html


def get_gentle_visualization_from_words(
    words: List[Any],  # List of WordTimestamp objects
    audio_file: Optional[Union[str, Path]] = None,
    title: str = "TorchAudio Aligner - Alignment Visualization",
) -> str:
    """
    Generate Gentle-style HTML visualization from WordTimestamp list.

    This is the simplified version that works with the new WordTimestamp API.
    Times are already in seconds, no frame conversion needed.

    Args:
        words: List of WordTimestamp objects (with text, start, end in seconds)
        audio_file: Path to audio file (optional)
        title: Page title

    Returns:
        HTML content string
    """
    # Build word data from WordTimestamp objects
    words_data = []
    for word in words:
        # Use original form if available, otherwise normalized text
        display_text = word.original if word.original else word.text
        word_info = {
            "word": display_text,
            "index": word.index,
            "aligned": True,  # All words in the list are aligned
            "start": word.start_seconds(),
            "end": word.end_seconds(),
        }
        words_data.append(word_info)

    # Build HTML
    html = _build_gentle_html(words_data, audio_file, title)
    return html


def save_gentle_html(
    word_alignment: Dict[int, Any],
    text: str,
    output_path: Union[str, Path],
    audio_file: Optional[Union[str, Path]] = None,
    frame_duration: float = 0.02,
    i_word_start: int = 0,
    i_word_end: Optional[int] = None,
    title: str = "TorchAudio Aligner - Alignment Visualization",
) -> str:
    """
    Save Gentle-style HTML visualization to file.

    Args:
        word_alignment: Dict mapping word index to AlignedWord
        text: Original text
        output_path: Path to save HTML file
        audio_file: Path to audio file
        frame_duration: Frame duration in seconds
        i_word_start: Start word index
        i_word_end: End word index
        title: Page title

    Returns:
        Path to saved file

    Example:
        >>> path = save_gentle_html(
        ...     result.word_alignments,
        ...     text,
        ...     "visualization.html",
        ...     audio_file="audio.mp3"
        ... )
        >>> print(f"Open {path} in a browser to view")
    """
    output_path = Path(output_path)
    html = get_gentle_visualization(
        word_alignment,
        text,
        audio_file=audio_file,
        frame_duration=frame_duration,
        i_word_start=i_word_start,
        i_word_end=i_word_end,
        title=title,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return str(output_path)


def create_interactive_demo(
    word_alignment: Dict[int, Any],
    text: str,
    audio_file: Union[str, Path],
    output_dir: Union[str, Path],
    frame_duration: float = 0.02,
    title: str = "TorchAudio Aligner - Interactive Demo",
) -> str:
    """
    Create an interactive HTML demo with pre-extracted audio clips.

    This creates a self-contained demo directory with:
    - index.html: Interactive visualization
    - audio_clips/: Directory with individual word audio clips (WAV)
    - full_audio.mp3: Copy of the original audio

    Features:
    - Segment selector (choose range or random)
    - Click on any word to play its audio clip
    - Works offline (no need for the original audio path)

    Args:
        word_alignment: Dict mapping word index to AlignedWord
        text: Original text
        audio_file: Path to audio file
        output_dir: Directory to save demo files
        frame_duration: Frame duration in seconds
        title: Page title

    Returns:
        Path to the index.html file

    Example:
        >>> path = create_interactive_demo(
        ...     result.word_alignments,
        ...     text,
        ...     "audio.mp3",
        ...     "demo_output/",
        ... )
        >>> print(f"Open {path} in a browser")
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("pydub required for interactive demo: pip install pydub")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create audio clips directory
    clips_dir = output_dir / "audio_clips"
    clips_dir.mkdir(exist_ok=True)

    # Load audio
    audio = AudioSegment.from_file(str(audio_file))

    # Process text
    text_words = text.split()
    total_words = len(text_words)

    # Build word data and extract audio clips
    words_data = []
    for i, word in enumerate(text_words):
        word_info = {
            "word": word,
            "index": i,
            "aligned": False,
            "start": None,
            "end": None,
            "clip": None,
        }

        if i in word_alignment:
            aligned = word_alignment[i]
            start = aligned.start_seconds(frame_duration)
            end = aligned.end_seconds(frame_duration)

            word_info["aligned"] = True
            word_info["start"] = start
            word_info["end"] = end

            # Extract and save audio clip
            clip_path = clips_dir / f"word_{i:05d}.wav"
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)

            # Add small padding
            start_ms = max(0, start_ms - 50)
            end_ms = min(len(audio), end_ms + 50)

            clip = audio[start_ms:end_ms]
            clip.export(str(clip_path), format="wav")

            word_info["clip"] = f"audio_clips/word_{i:05d}.wav"

        words_data.append(word_info)

    # Build HTML with segment selector
    words_json = json.dumps(words_data)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        h1 {{
            color: white;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            margin-bottom: 20px;
        }}
        .controls {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            align-items: center;
        }}
        .control-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        label {{
            font-weight: 600;
            color: #333;
        }}
        input[type="number"] {{
            width: 80px;
            padding: 8px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
        }}
        input[type="number"]:focus {{
            border-color: #667eea;
            outline: none;
        }}
        button {{
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}
        button.primary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        button.primary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
        button.secondary {{
            background: #f0f0f0;
            color: #333;
        }}
        button.secondary:hover {{
            background: #e0e0e0;
        }}
        .stats {{
            background: white;
            padding: 15px 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
        }}
        .stat {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: 700;
            color: #667eea;
        }}
        .stat-label {{
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
        }}
        .transcript {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            line-height: 2.2;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            max-height: 60vh;
            overflow-y: auto;
        }}
        .word {{
            display: inline;
            cursor: pointer;
            padding: 4px 6px;
            margin: 2px;
            border-radius: 4px;
            transition: all 0.2s;
            font-size: 16px;
        }}
        .word.aligned {{
            background: #e8f5e9;
            color: #2e7d32;
        }}
        .word.aligned:hover {{
            background: #c8e6c9;
            transform: scale(1.05);
        }}
        .word.aligned.playing {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            transform: scale(1.1);
        }}
        .word.unaligned {{
            background: #ffebee;
            color: #c62828;
        }}
        .word.hidden {{
            display: none;
        }}
        .legend {{
            margin-top: 20px;
            padding: 15px;
            background: rgba(255,255,255,0.9);
            border-radius: 8px;
            font-size: 14px;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
        }}
        .legend-item span {{
            padding: 3px 8px;
            border-radius: 4px;
            margin-right: 5px;
        }}
        .current-segment {{
            color: white;
            font-size: 14px;
            margin-left: auto;
        }}
        @media (max-width: 768px) {{
            .controls {{
                flex-direction: column;
                align-items: stretch;
            }}
            .control-group {{
                justify-content: space-between;
            }}
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>

    <div class="controls">
        <div class="control-group">
            <label>Start:</label>
            <input type="number" id="startIdx" value="0" min="0" max="{total_words - 1}">
        </div>
        <div class="control-group">
            <label>End:</label>
            <input type="number" id="endIdx" value="{min(100, total_words)}" min="1" max="{total_words}">
        </div>
        <button class="primary" onclick="showSegment()">Show Segment</button>
        <button class="secondary" onclick="showRandom()">Random 50 Words</button>
        <button class="secondary" onclick="showAll()">Show All</button>
        <div class="current-segment" id="currentSegment">Showing words 0-{min(100, total_words)}</div>
    </div>

    <div class="stats">
        <div class="stat">
            <div class="stat-value">{total_words}</div>
            <div class="stat-label">Total Words</div>
        </div>
        <div class="stat">
            <div class="stat-value">{sum(1 for w in words_data if w['aligned'])}</div>
            <div class="stat-label">Aligned</div>
        </div>
        <div class="stat">
            <div class="stat-value">{sum(1 for w in words_data if not w['aligned'])}</div>
            <div class="stat-label">Unaligned</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="visibleCount">{min(100, total_words)}</div>
            <div class="stat-label">Currently Shown</div>
        </div>
    </div>

    <div class="transcript" id="transcript">
    </div>

    <div class="legend">
        <div class="legend-item">
            <span class="word aligned">Aligned</span> Click to play audio
        </div>
        <div class="legend-item">
            <span class="word unaligned">Unaligned</span> No audio available
        </div>
    </div>

    <script>
        const wordsData = {words_json};
        const totalWords = {total_words};
        let currentAudio = null;

        // Build transcript
        function buildTranscript() {{
            const container = document.getElementById('transcript');
            container.innerHTML = '';

            wordsData.forEach((w, i) => {{
                const span = document.createElement('span');
                span.className = 'word ' + (w.aligned ? 'aligned' : 'unaligned');
                span.textContent = w.word;
                span.dataset.index = i;

                if (w.aligned && w.clip) {{
                    span.onclick = () => playWord(i);
                }}

                container.appendChild(span);
                container.appendChild(document.createTextNode(' '));
            }});
        }}

        function playWord(index) {{
            const w = wordsData[index];
            if (!w.clip) return;

            // Stop current audio
            if (currentAudio) {{
                currentAudio.pause();
                currentAudio = null;
            }}

            // Remove playing class from all
            document.querySelectorAll('.word.playing').forEach(el => el.classList.remove('playing'));

            // Play new audio
            currentAudio = new Audio(w.clip);
            currentAudio.play();

            // Highlight word
            const wordEl = document.querySelector(`.word[data-index="${{index}}"]`);
            if (wordEl) {{
                wordEl.classList.add('playing');
                currentAudio.onended = () => wordEl.classList.remove('playing');
            }}
        }}

        function showSegment() {{
            const start = parseInt(document.getElementById('startIdx').value);
            const end = parseInt(document.getElementById('endIdx').value);
            filterWords(start, end);
        }}

        function showRandom() {{
            const numWords = 50;
            const maxStart = Math.max(0, totalWords - numWords);
            const start = Math.floor(Math.random() * maxStart);
            const end = Math.min(start + numWords, totalWords);

            document.getElementById('startIdx').value = start;
            document.getElementById('endIdx').value = end;
            filterWords(start, end);
        }}

        function showAll() {{
            document.getElementById('startIdx').value = 0;
            document.getElementById('endIdx').value = totalWords;
            filterWords(0, totalWords);
        }}

        function filterWords(start, end) {{
            const words = document.querySelectorAll('.word');
            let visibleCount = 0;

            words.forEach((w, i) => {{
                if (i >= start && i < end) {{
                    w.classList.remove('hidden');
                    visibleCount++;
                }} else {{
                    w.classList.add('hidden');
                }}
            }});

            document.getElementById('currentSegment').textContent = `Showing words ${{start}}-${{end}}`;
            document.getElementById('visibleCount').textContent = visibleCount;
        }}

        // Initialize
        buildTranscript();
        filterWords(0, Math.min(100, totalWords));
    </script>
</body>
</html>"""

    # Save HTML
    html_path = output_dir / "index.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Interactive demo created at: {output_dir}")
    print(f"  - index.html: Open this in a browser")
    print(f"  - audio_clips/: Contains {sum(1 for w in words_data if w['aligned'])} word audio clips")

    return str(html_path)
