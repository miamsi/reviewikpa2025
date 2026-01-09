import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="IKPA 2025 - Professional Audit Dashboard", layout="wide")

# --- KONFIGURASI DATA ---
FILE_PATH = r"C:\Users\michael.sidabutar\Documents\analisis ikpa 2025\data.csv"

# Bobot Komponen IKPA Resmi
WEIGHTS = {
    'Revisi DIPA': 10, 'Deviasi Halaman III DIPA': 15, 'Penyerapan Anggaran': 20,
    'Belanja Kontraktual': 10, 'Penyelesaian Tagihan': 10, 'Pengelolaan UP dan TUP': 10,
    'Capaian Output': 25
}

@st.cache_data
def load_and_process():
    try:
        df = pd.read_csv(FILE_PATH)
    except:
        df = pd.read_csv('data.csv')

    def clean_numeric(val):
        if pd.isna(val) or val == '': return 0.0
        val = str(val).replace('%', '').replace(',', '.')
        try: return float(val)
        except: return 0.0

    cols_to_clean = list(WEIGHTS.keys()) + ['Nilai Akhir (Nilai Total/Konversi Bobot)', 'Konversi Bobot']
    for col in cols_to_clean:
        df[col] = df[col].apply(clean_numeric)

    df['Satker_Full'] = df['Uraian Satker'] + " (" + df['Kode Satker'].astype(str) + ")"
    
    # Logika Deteksi Komponen 'Real 0' (Gagal tapi dihitung)
    def detect_real_zeros(row):
        nonzero_comps = [k for k in WEIGHTS.keys() if row[k] > 0]
        sum_nonzero_weights = sum(WEIGHTS[k] for k in nonzero_comps)
        diff = round(row['Konversi Bobot'] - sum_nonzero_weights, 2)
        
        real_zeros = []
        if diff > 0:
            zero_comps = [k for k in WEIGHTS.keys() if row[k] == 0]
            temp_diff = diff
            for k in zero_comps:
                if WEIGHTS[k] <= temp_diff + 0.05:
                    real_zeros.append(k)
                    temp_diff -= WEIGHTS[k]
                if temp_diff <= 0: break
        return real_zeros

    df['Real_Zeros'] = df.apply(detect_real_zeros, axis=1)

    def get_labels(row):
        active_vals = {k: row[k] for k in WEIGHTS.keys() if row[k] > 0}
        strongest = max(active_vals, key=active_vals.get) if active_vals else "N/A"
        
        if row['Real_Zeros']:
            weakest = f"{', '.join(row['Real_Zeros'])} (SKOR 0)"
        elif active_vals:
            w_comp = min(active_vals, key=active_vals.get)
            weakest = "-" if active_vals[w_comp] == 100 else w_comp
        else:
            weakest = "N/A"
        return strongest, weakest

    df[['Komponen Terbaik', 'Komponen Terlemah']] = df.apply(
        lambda x: pd.Series(get_labels(x)), axis=1
    )
    return df

df = load_and_process()

# --- FUNGSI STYLING (HIGHLIGHT) ---
def apply_audit_style(df_to_style):
    def style_logic(row):
        styles = [''] * len(row)
        bad_list = row.get('Real_Zeros', [])
        for i, col_name in enumerate(row.index):
            # Merah Pekat untuk Gagal Total (Real 0)
            if col_name in bad_list:
                styles[i] = 'background-color: #ff4b4b; color: white; font-weight: bold'
            # Merah Muda untuk Performa Rendah (<90)
            elif col_name in WEIGHTS.keys() and 0 < row[col_name] < 90:
                styles[i] = 'background-color: #fff4f4; color: #990000'
        return styles

    return df_to_style.style.apply(style_logic, axis=1).hide(['Real_Zeros'], axis='columns')

# --- UI UTAMA ---
st.title("🏛️ Portal Strategis Monitoring IKPA 2025")
st.markdown(f"**Audit Mode:** :red[Real-Zero Detection Active] | File: `{FILE_PATH}`")

list_pic = sorted(df['PIC'].unique())
tabs = st.tabs([f"👤 PIC {p}" for p in list_pic])

for i, tab in enumerate(tabs):
    pic_id = list_pic[i]
    with tab:
        pic_df = df[df['PIC'] == pic_id].copy()
        
        # 1. Dashboard Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rerata Skor Unit", f"{pic_df['Nilai Akhir (Nilai Total/Konversi Bobot)'].mean():.2f}")
        m2.metric("Total Unit Binaan", len(pic_df))
        m3.metric("Lulus Sempurna", len(pic_df[pic_df['Nilai Akhir (Nilai Total/Konversi Bobot)'] == 100]))
        m4.metric("Atensi (<90)", len(pic_df[pic_df['Nilai Akhir (Nilai Total/Konversi Bobot)'] < 90]))

        st.divider()

        # 2. Insights: Ranking KPI (Komponen)
        st.subheader("📊 Analisis Performa Komponen (KPI)")
        avg_kpi = pic_df[list(WEIGHTS.keys())].mean().sort_values(ascending=False)
        c_kpi1, c_kpi2 = st.columns([1.5, 1])
        with c_kpi1:
            fig_kpi = px.bar(avg_kpi, orientation='h', labels={'value':'Rerata Skor', 'index':'Komponen'}, 
                             color=avg_kpi.values, color_continuous_scale='RdYlGn')
            fig_kpi.update_layout(showlegend=False, height=300, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_kpi, use_container_width=True)
        with c_kpi2:
            st.write("**Top & Bottom KPI:**")
            st.dataframe(avg_kpi.rename("Rerata Skor"), use_container_width=True)

        st.divider()

        # 3. Panggung Juara & Zona Merah (Top/Bottom 10 Satker)
        st.subheader("🏆 Peringkat Performa Unit")
        col_t, col_b = st.columns(2)
        view_cols = ['Satker_Full', 'Nilai Akhir (Nilai Total/Konversi Bobot)'] + list(WEIGHTS.keys()) + ['Real_Zeros']
        
        with col_t:
            st.success("🥇 10 Unit Performa Terbaik")
            top_df = pic_df.sort_values('Nilai Akhir (Nilai Total/Konversi Bobot)', ascending=False).head(10)[view_cols]
            st.dataframe(apply_audit_style(top_df), hide_index=True)

        with col_b:
            st.error("🚨 10 Unit Performa Terendah")
            bottom_df = pic_df.sort_values('Nilai Akhir (Nilai Total/Konversi Bobot)', ascending=True).head(10)[view_cols]
            st.dataframe(apply_audit_style(bottom_df), hide_index=True)

        st.divider()

        # 4. Masalah Terbanyak (Individual Component Basis)
        st.subheader("⚠️ Komponen Paling Bermasalah (Daftar Unit)")
        
        # Flatten semua kelemahan (Real Zeros + Skor Terendah biasa)
        all_bad_comps = [comp for sublist in pic_df['Real_Zeros'] for comp in sublist]
        other_weak = pic_df[pic_df['Real_Zeros'].map(len) == 0]['Komponen Terlemah']
        all_weaknesses = all_bad_comps + other_weak[~other_weak.isin(['-', 'N/A'])].tolist()
        
        if all_weaknesses:
            weak_freq = pd.Series(all_weaknesses).value_counts().head(3)
            for comp_name, count in weak_freq.items():
                with st.expander(f"🚩 Masalah Utama: {comp_name} ({count} Unit Terdampak)", expanded=True):
                    # Filter unit yang memiliki komponen ini di Real_Zeros ATAU sebagai Komponen Terlemah
                    mask = pic_df['Real_Zeros'].apply(lambda x: comp_name in x) | (pic_df['Komponen Terlemah'].str.contains(comp_name))
                    affected = pic_df[mask][view_cols].sort_values('Nilai Akhir (Nilai Total/Konversi Bobot)')
                    st.dataframe(apply_audit_style(affected), hide_index=True, use_container_width=True)
        else:
            st.success("Tidak ada masalah signifikan terdeteksi.")

        st.divider()

        # 5. Visualisasi Radar (Selalu Terbuka)
        st.subheader("🎯 Profil Detail Per Unit")
        c_rad1, c_rad2 = st.columns([1, 2.5])
        with c_rad1:
            sel_u = st.selectbox("Cari Satker:", pic_df['Satker_Full'].unique(), key=f"sel_{pic_id}")
            u_row = pic_df[pic_df['Satker_Full'] == sel_u].iloc[0]
            st.metric("Skor Akhir", f"{u_row['Nilai Akhir (Nilai Total/Konversi Bobot)']:.2f}")
            st.markdown(f"**Kekuatan:** :green[{u_row['Komponen Terbaik']}]")
            st.markdown(f"**Kelemahan Utama:** :red[{u_row['Komponen Terlemah']}]")
        with c_rad2:
            fig_rad = px.line_polar(r=u_row[list(WEIGHTS.keys())].values, theta=list(WEIGHTS.keys()), line_close=True, range_r=[0,100])
            fig_rad.update_traces(fill='toself', line_color='#d62728' if u_row['Real_Zeros'] else '#1f77b4')
            fig_rad.update_layout(height=400, margin=dict(l=80, r=80, t=20, b=20))
            st.plotly_chart(fig_rad, use_container_width=True)

        st.divider()

        # 6. Tabel Audit Lengkap
        st.subheader("📋 Seluruh Data Audit")
        full_audit_cols = ['Satker_Full', 'Nilai Akhir (Nilai Total/Konversi Bobot)'] + list(WEIGHTS.keys()) + ['Komponen Terbaik', 'Komponen Terlemah', 'Real_Zeros']
        st.dataframe(
            apply_audit_style(pic_df[full_audit_cols].sort_values('Nilai Akhir (Nilai Total/Konversi Bobot)')),
            hide_index=True, use_container_width=True
        )