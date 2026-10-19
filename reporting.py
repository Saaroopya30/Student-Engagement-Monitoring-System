from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def generate_reports(records: List[Dict[str, object]], output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    label_counts = Counter(record["label"] for record in records)
    per_frame = defaultdict(Counter)
    per_student = defaultdict(Counter)

    for record in records:
        per_frame[record["frame_index"]][record["label"]] += 1
        per_student[record["track_id"]][record["label"]] += 1

    total_records = len(records)
    overall_engaged_pct = round(
        (label_counts.get("Engaged", 0) / total_records) * 100, 2
    ) if total_records else 0.0
    attention_score_pct = round(
        (
            (
                label_counts.get("Engaged", 0)
                + 0.5 * label_counts.get("Partially Engaged", 0)
            )
            / total_records
        )
        * 100,
        2,
    ) if total_records else 0.0

    student_percentages = {}
    for track_id, counts in per_student.items():
        student_total = sum(counts.values())
        student_percentages[str(track_id)] = {
            "engaged_percentage": round((counts["Engaged"] / student_total) * 100, 2)
            if student_total
            else 0.0,
            "attention_score_percentage": round(
                ((counts["Engaged"] + 0.5 * counts["Partially Engaged"]) / student_total) * 100,
                2,
            )
            if student_total
            else 0.0,
            "total_detections": student_total,
        }

    summary = {
        "total_records": len(records),
        "label_counts": dict(label_counts),
        "student_count": len(per_student),
        "students": {str(track_id): dict(counts) for track_id, counts in per_student.items()},
        "overall_engaged_percentage": overall_engaged_pct,
        "overall_attention_score_percentage": attention_score_pct,
        "student_percentages": student_percentages,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    student_summary_path = output_dir / "student_summary.csv"
    with student_summary_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            "track_id,engaged,partially_engaged,not_engaged,engaged_percentage,"
            "attention_score_percentage,dominant_label\n"
        )
        for track_id, counts in sorted(per_student.items()):
            dominant_label = counts.most_common(1)[0][0] if counts else "Unknown"
            percentages = student_percentages[str(track_id)]
            handle.write(
                f"{track_id},{counts['Engaged']},{counts['Partially Engaged']},"
                f"{counts['Not Engaged']},{percentages['engaged_percentage']},"
                f"{percentages['attention_score_percentage']},{dominant_label}\n"
            )

    if records:
        frames = sorted(per_frame.keys())
        engaged = [per_frame[frame]["Engaged"] for frame in frames]
        partial = [per_frame[frame]["Partially Engaged"] for frame in frames]
        not_engaged = [per_frame[frame]["Not Engaged"] for frame in frames]

        plt.figure(figsize=(12, 6))
        plt.plot(frames, engaged, label="Engaged", color="green")
        plt.plot(frames, partial, label="Partially Engaged", color="orange")
        plt.plot(frames, not_engaged, label="Not Engaged", color="red")
        plt.title("Engagement Trend Over Time")
        plt.xlabel("Frame")
        plt.ylabel("Detected Students")
        plt.legend()
        plt.tight_layout()
        trend_path = output_dir / "engagement_trend.png"
        plt.savefig(trend_path)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.bar(label_counts.keys(), label_counts.values(), color=["orange", "green", "red"])
        plt.title("Overall Engagement Distribution")
        plt.ylabel("Detections")
        plt.tight_layout()
        distribution_path = output_dir / "engagement_distribution.png"
        plt.savefig(distribution_path)
        plt.close()
    else:
        trend_path = output_dir / "engagement_trend.png"
        distribution_path = output_dir / "engagement_distribution.png"

    dashboard_path = output_dir / "dashboard.html"
    dashboard_path.write_text(
        f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Student Engagement Dashboard</title>
  <style>
    body {{
      font-family: "Segoe UI", sans-serif;
      margin: 0;
      padding: 32px;
      background: linear-gradient(135deg, #f7f4ea 0%, #dde8f5 100%);
      color: #1f2933;
    }}
    h1, h2 {{ margin: 0 0 12px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .card {{
      background: rgba(255, 255, 255, 0.88);
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 10px 30px rgba(31, 41, 51, 0.08);
    }}
    img {{
      max-width: 100%;
      border-radius: 14px;
      margin-top: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: rgba(255, 255, 255, 0.9);
      border-radius: 14px;
      overflow: hidden;
    }}
    th, td {{
      padding: 12px;
      border-bottom: 1px solid #d9e2ec;
      text-align: left;
    }}
    th {{ background: #102a43; color: white; }}
  </style>
</head>
<body>
  <h1>Student Engagement Dashboard</h1>
  <p>Generated classroom analytics based on the processed input video.</p>
    <div class="grid">
    <div class="card"><h2>Total Records</h2><p>{summary['total_records']}</p></div>
    <div class="card"><h2>Students Tracked</h2><p>{summary['student_count']}</p></div>
    <div class="card"><h2>Engaged %</h2><p>{summary['overall_engaged_percentage']}%</p></div>
    <div class="card"><h2>Attention Score %</h2><p>{summary['overall_attention_score_percentage']}%</p></div>
    <div class="card"><h2>Engaged</h2><p>{label_counts.get('Engaged', 0)}</p></div>
    <div class="card"><h2>Partially Engaged</h2><p>{label_counts.get('Partially Engaged', 0)}</p></div>
    <div class="card"><h2>Not Engaged</h2><p>{label_counts.get('Not Engaged', 0)}</p></div>
  </div>
  <div class="grid">
    <div class="card">
      <h2>Trend Report</h2>
      <img src="engagement_trend.png" alt="Engagement trend chart">
    </div>
    <div class="card">
      <h2>Distribution Report</h2>
      <img src="engagement_distribution.png" alt="Engagement distribution chart">
    </div>
  </div>
  <div class="card">
    <h2>Per Student Summary</h2>
    <table>
      <thead>
        <tr>
          <th>Track ID</th>
          <th>Engaged</th>
          <th>Partially Engaged</th>
          <th>Not Engaged</th>
          <th>Engaged %</th>
          <th>Attention Score %</th>
          <th>Dominant Label</th>
        </tr>
      </thead>
      <tbody>
        {''.join(
            f"<tr><td>{track_id}</td><td>{counts['Engaged']}</td><td>{counts['Partially Engaged']}</td>"
            f"<td>{counts['Not Engaged']}</td><td>{student_percentages[str(track_id)]['engaged_percentage']}%</td>"
            f"<td>{student_percentages[str(track_id)]['attention_score_percentage']}%</td>"
            f"<td>{counts.most_common(1)[0][0]}</td></tr>"
            for track_id, counts in sorted(per_student.items())
        )}
      </tbody>
    </table>
  </div>
</body>
</html>
        """.strip(),
        encoding="utf-8",
    )

    return {
        "summary_json": summary_path,
        "trend_chart": trend_path,
        "distribution_chart": distribution_path,
        "student_summary_csv": student_summary_path,
        "dashboard_html": dashboard_path,
    }
