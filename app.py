import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

st.title("🏎️ Formula Telemetry Decision System")

# =========================
# CSVアップロード
# =========================
file = st.file_uploader("CSVアップロード", type="csv")

if file:

    df = pd.read_csv(file, skiprows=1)

    # =========================
    # 前処理
    # =========================
    cols = ["AccX","AccY","AccZ","GyroX","GyroY","GyroZ","MPa1","MPa2"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna()

    df["dP"] = df["MPa2"] - df["MPa1"]
    df["latG"] = df["AccY"]
    df["yaw"] = df["GyroZ"]

    # =========================
    # ラップ分割（簡易）
    # =========================
    df["AccX_smooth"] = df["AccX"].rolling(10, center=True).mean()

    peaks, _ = find_peaks(df["AccX_smooth"], distance=100)

    laps = []
    for i in range(len(peaks)-1):
        laps.append(df.iloc[peaks[i]:peaks[i+1]])

    st.write("検出ラップ数:", len(laps))

    if len(laps) >= 2:

        lap1 = laps[-2]
        lap2 = laps[-1]

        # =========================
        # コーナー分割
        # =========================
        def analyze_lap(lap_df):

            gyro = lap_df["yaw"].rolling(5).mean()
            threshold = np.std(gyro) * 1.5

            peaks, _ = find_peaks(np.abs(gyro), height=threshold, distance=50)

            segments = []
            prev = 0

            for i, p in enumerate(peaks):
                seg = lap_df.iloc[prev:p]

                segments.append({
                    "segment_id": i,
                    "dP_mean": seg["dP"].mean(),
                    "latG_mean": seg["latG"].mean()
                })

                prev = p

            return segments

        lap1_feat = analyze_lap(lap1)
        lap2_feat = analyze_lap(lap2)

        # =========================
        # diff生成
        # =========================
        min_len = min(len(lap1_feat), len(lap2_feat))

        diff = []

        for i in range(min_len):
            f1 = lap1_feat[i]
            f2 = lap2_feat[i]

            diff.append({
                "segment_id": i,
                "dP_diff": f2["dP_mean"] - f1["dP_mean"],
                "latG_diff": f2["latG_mean"] - f1["latG_mean"]
            })

        st.subheader("📊 差分データ")
        st.write(pd.DataFrame(diff))

        # =========================
        # 意思決定（完成版）
        # =========================
        st.subheader("🧠 分析結果")

        total_latG = sum([d["latG_diff"] for d in diff])
        total_dP = sum([d["dP_diff"] for d in diff])

        # 総合
        st.markdown("### 総合評価")

        grip_flag = False
        aero_flag = False

        if total_latG < -0.03:
            st.error("グリップ低下（タイヤ）")
            grip_flag = True

        if abs(total_dP) > 0.3:
            st.warning("空力は安定していない")
            aero_flag = True

        # 最重要区間
        scores = [abs(d["latG_diff"]) + abs(d["dP_diff"]) for d in diff]
        worst_idx = scores.index(max(scores))

        st.markdown("### 最重要区間")
        st.write(f"区間 {worst_idx}")

        # 詳細
        st.markdown("### 詳細")

        for i, d in enumerate(diff):
            if i == worst_idx:

                if d["latG_diff"] < -0.01:
                    st.write(f"区間{i}：グリップ低下")

                if d["dP_diff"] < -0.2:
                    st.write(f"区間{i}：空力低下")

                elif d["dP_diff"] > 0.2:
                    st.write(f"区間{i}：空力改善")

        # 結論
        st.markdown("### 結論")

        if grip_flag and aero_flag:
            st.success("タイヤは悪化傾向 / 空力は安定していない")

        elif grip_flag:
            st.success("タイヤ劣化が主原因")

        elif aero_flag:
            st.success("空力の影響が大きい")

        else:
            st.success("大きな問題なし（安定）")

# save point
# before lap detection fix
