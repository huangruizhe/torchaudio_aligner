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
