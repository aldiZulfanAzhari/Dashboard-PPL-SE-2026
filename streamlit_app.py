import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             silhouette_samples, silhouette_score)
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Dashboard PPL SE2026", page_icon="📊", layout="wide")

ATRIBUT = ["rasio_penyelesaian", "rata_harian", "persen_hari_aktif", "volatilitas"]
LABEL_ATRIBUT = ["Rasio penyelesaian", "Rata-rata harian (log)", "% hari aktif", "Volatilitas (log)"]
WARNA = {1: "#2a9d8f", 2: "#e9c46a", 3: "#e76f51"}
KOLOM_WAJIB = ["Nama Petugas", "TOTAL", "Target", "Status"]
FILE_DEFAULT = ["data.csv", "laporan_progress_petugas.csv"]


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def baca_csv(sumber):
    return pd.read_csv(sumber)


@st.cache_data(show_spinner=False)
def siapkan_data(raw: pd.DataFrame, batas_rasio: float):
    """Kolom tanggal (dd/mm) = laporan kumulatif -> diubah jadi laporan baru per hari."""
    kolom_tanggal = [c for c in raw.columns if "/" in c]
    kum = raw[kolom_tanggal].astype(float)

    kum_isi = kum.replace(0, np.nan).ffill(axis=1).fillna(0)
    harian = kum_isi.diff(axis=1)
    harian.iloc[:, 0] = kum_isi.iloc[:, 0]

    cek = {
        "n_petugas": len(raw),
        "n_hari": len(kolom_tanggal),
        "rentang": f"{kolom_tanggal[0]} s.d. {kolom_tanggal[-1]}",
        "total_sama_maks": int((raw["TOTAL"] == kum.max(axis=1)).sum()),
        "selisih_negatif": int((harian < 0).sum().sum()),
        "total_cocok": int((harian.sum(axis=1) == raw["TOTAL"]).sum()),
        "nilai_kosong": int(raw[KOLOM_WAJIB + kolom_tanggal].isna().sum().sum()),
        "target_nol": int((raw["Target"] == 0).sum()),
    }

    n_hari = len(kolom_tanggal)
    df = raw[KOLOM_WAJIB].copy()
    df["rasio_penyelesaian"] = (raw["TOTAL"] / raw["Target"].replace(0, np.nan)).fillna(0).clip(upper=batas_rasio)
    df["rata_harian"] = harian.sum(axis=1) / n_hari
    df["hari_aktif"] = (harian > 0).sum(axis=1)
    df["persen_hari_aktif"] = df["hari_aktif"] / n_hari * 100
    df["volatilitas"] = harian.std(axis=1)
    return df, cek


def normalisasi(aktif: pd.DataFrame):
    data = aktif[ATRIBUT].copy()
    data["rata_harian"] = np.log1p(data["rata_harian"])
    data["volatilitas"] = np.log1p(data["volatilitas"])
    korelasi = data.corr().round(3)
    Z = MinMaxScaler().fit_transform(data)
    return Z, korelasi


@st.cache_data(show_spinner=False)
def evaluasi_k(Z: np.ndarray, k_min: int, k_max: int, n_init: int, seed: int):
    baris = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=seed).fit(Z)
        baris.append({
            "k": k,
            "WCSS": km.inertia_,
            "Silhouette": silhouette_score(Z, km.labels_),
            "Davies-Bouldin": davies_bouldin_score(Z, km.labels_),
            "Ukuran klaster": str(sorted(np.bincount(km.labels_).tolist(), reverse=True)),
        })
    ev = pd.DataFrame(baris)
    ev["Penurunan WCSS"] = -ev["WCSS"].diff()
    return ev


def klasterkan(aktif: pd.DataFrame, Z: np.ndarray, k: int, n_init: int, seed: int):
    km = KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=seed).fit(Z)
    aktif = aktif.copy()
    aktif["klaster_mentah"] = km.labels_
    urutan = (aktif.groupby("klaster_mentah")["rasio_penyelesaian"].mean()
              .sort_values(ascending=False).index.tolist())
    peta = {lama: baru + 1 for baru, lama in enumerate(urutan)}
    aktif["klaster"] = aktif["klaster_mentah"].map(peta)
    nama = {1: "Tinggi", 2: "Sedang", 3: "Rendah"} if k == 3 else {i: f"Klaster {i}" for i in range(1, k + 1)}
    aktif["kelompok"] = aktif["klaster"].map(lambda c: f"Klaster {c} ({nama[c]})" if k == 3 else nama[c])
    aktif["silhouette"] = silhouette_samples(Z, km.labels_)
    return aktif, km, nama


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Pengaturan")

unggahan = st.sidebar.file_uploader("Unggah CSV laporan progres petugas", type="csv")
sumber = None
if unggahan is not None:
    sumber = unggahan
else:
    for f in FILE_DEFAULT:
        if os.path.exists(f):
            sumber = f
            st.sidebar.caption(f"Memakai berkas bawaan repo: `{f}`")
            break

st.sidebar.subheader("Parameter K-Means")
k_final = st.sidebar.slider("Jumlah klaster (k)", 2, 7, 3)
k_min, k_max = 2, 7
n_init = st.sidebar.number_input("n_init", 1, 50, 10)
seed = st.sidebar.number_input("random_state", 0, 9999, 42)
batas_rasio = st.sidebar.number_input("Batas maksimum rasio penyelesaian", 0.1, 5.0, 1.0, 0.1)
n_seed_uji = st.sidebar.slider("Jumlah seed uji kestabilan", 5, 50, 20)

st.sidebar.subheader("Privasi")
sembunyikan_nama = st.sidebar.checkbox("Samarkan nama petugas", value=True,
                                       help="Nama diganti 'Petugas <no. urut>' pada tabel dan unduhan.")

# ---------------------------------------------------------------- header
st.title("Dashboard Klasterisasi Kinerja Petugas PPL SE2026")
st.caption("BPS Kabupaten Indragiri Hilir · Algoritma K-Means · data SIPANTAU")

if sumber is None:
    st.info("Unggah berkas CSV laporan progres petugas pada panel kiri untuk memulai.")
    st.stop()

raw = baca_csv(sumber)

kurang = [c for c in KOLOM_WAJIB if c not in raw.columns]
if kurang or not any("/" in c for c in raw.columns):
    st.error(f"Format CSV tidak sesuai. Kolom wajib: {', '.join(KOLOM_WAJIB)} + kolom tanggal berformat dd/mm. "
             f"Kolom yang hilang: {', '.join(kurang) if kurang else 'kolom tanggal'}.")
    st.stop()

raw = raw.reset_index(drop=True)
df, cek = siapkan_data(raw, batas_rasio)
if sembunyikan_nama:
    df["Nama Petugas"] = [f"Petugas {i + 1}" for i in range(len(df))]

tidak_aktif = df[df["hari_aktif"] == 0].copy()
aktif = df[df["hari_aktif"] > 0].copy().reset_index(drop=True)

if len(aktif) <= k_final:
    st.error("Jumlah petugas yang pernah melapor terlalu sedikit untuk jumlah klaster yang dipilih.")
    st.stop()

Z, korelasi = normalisasi(aktif)
aktif, km, nama_tingkat = klasterkan(aktif, Z, k_final, int(n_init), int(seed))
sil = silhouette_score(Z, km.labels_)
dbi = davies_bouldin_score(Z, km.labels_)

# ---------------------------------------------------------------- ringkasan
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total petugas", len(df))
c2.metric("Pernah melapor", len(aktif), f"{len(aktif) / len(df) * 100:.1f}%", delta_color="off")
c3.metric("Tanpa laporan", len(tidak_aktif), f"{len(tidak_aktif) / len(df) * 100:.1f}%", delta_color="off")
c4.metric("Silhouette", f"{sil:.3f}")
c5.metric("Davies-Bouldin", f"{dbi:.3f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Ringkasan data", "Jumlah klaster", "Hasil klaster", "Tabel petugas", "Kualitas data"])

# ---- tab 1
with tab1:
    st.subheader("Keaktifan pelaporan")
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    nilai = [len(aktif), len(tidak_aktif)]
    ax.bar(["Pernah melapor", "Tanpa laporan"], nilai, color=["#2a6f97", "#adb5bd"])
    for i, v in enumerate(nilai):
        ax.text(i, v + max(nilai) * 0.02, f"{v} ({v / len(df) * 100:.1f}%)", ha="center")
    ax.set_ylabel("Jumlah petugas")
    ax.set_ylim(0, max(nilai) * 1.15)
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("Status penyelesaian")
    st.dataframe(df["Status"].value_counts().rename("Jumlah").to_frame(), width='stretch')

    st.subheader("Statistik atribut (petugas yang pernah melapor)")
    st.dataframe(aktif[ATRIBUT].describe().round(3), width='stretch')

    st.subheader("Korelasi antaratribut (setelah log, sebelum normalisasi)")
    st.dataframe(korelasi, width='stretch')

# ---- tab 2
with tab2:
    st.subheader("Evaluasi jumlah klaster")
    ev = evaluasi_k(Z, k_min, k_max, int(n_init), int(seed))
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.plot(ev["k"], ev["WCSS"], marker="o", color="#2a6f97")
        ax.axvline(k_final, ls="--", color="#c1121f", label=f"k = {k_final}")
        ax.set_xlabel("Jumlah klaster (k)")
        ax.set_ylabel("WCSS")
        ax.set_title("Elbow Method")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    with col_b:
        fig, axs = plt.subplots(1, 2, figsize=(7, 3.6))
        axs[0].plot(ev["k"], ev["Silhouette"], marker="o", color="#2a9d8f")
        axs[0].set_title("Silhouette (besar = baik)", fontsize=9)
        axs[1].plot(ev["k"], ev["Davies-Bouldin"], marker="o", color="#e76f51")
        axs[1].set_title("Davies-Bouldin (kecil = baik)", fontsize=9)
        for a in axs:
            a.axvline(k_final, ls="--", color="#c1121f")
            a.set_xlabel("k")
            a.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    st.dataframe(ev.round(3), width='stretch', hide_index=True)
    st.caption("Ubah k pada panel kiri setelah membaca grafik evaluasi.")

# ---- tab 3
with tab3:
    st.subheader(f"Hasil K-Means (k = {k_final})")
    ukuran = aktif["kelompok"].value_counts().sort_index()
    st.dataframe(ukuran.rename("Jumlah petugas").to_frame(), width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        pca = PCA(n_components=2, random_state=int(seed))
        P = pca.fit_transform(Z)
        fig, ax = plt.subplots(figsize=(6, 4.4))
        for c in sorted(aktif["klaster"].unique()):
            m = (aktif["klaster"] == c).values
            ax.scatter(P[m, 0], P[m, 1], s=22, alpha=0.8, color=WARNA.get(c),
                       label=f"{aktif.loc[m, 'kelompok'].iloc[0]} (n={m.sum()})")
        pusat = pca.transform(km.cluster_centers_)
        ax.scatter(pusat[:, 0], pusat[:, 1], marker="X", s=150, c="black", label="Centroid")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% varians)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% varians)")
        ax.set_title("Sebaran klaster (proyeksi PCA)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    with col_b:
        Z_df = pd.DataFrame(Z, columns=ATRIBUT)
        Z_df["klaster"] = aktif["klaster"].values
        prof_norm = Z_df.groupby("klaster")[ATRIBUT].mean()
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        lebar = 0.8 / len(prof_norm)
        for i, (c, baris) in enumerate(prof_norm.iterrows()):
            ax.bar(np.arange(4) + i * lebar, baris.values, lebar, color=WARNA.get(c),
                   label=f"Klaster {c}" + (f" ({nama_tingkat[c]})" if k_final == 3 else ""))
        ax.set_xticks(np.arange(4) + lebar * (len(prof_norm) - 1) / 2)
        ax.set_xticklabels(LABEL_ATRIBUT, fontsize=7, rotation=10)
        ax.set_ylabel("Rata-rata ternormalisasi (0-1)")
        ax.set_title("Profil klaster")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    st.subheader("Profil klaster")
    profil = (aktif.groupby("kelompok")
              .agg(jumlah_petugas=("Nama Petugas", "count"),
                   rasio_penyelesaian=("rasio_penyelesaian", "mean"),
                   rata_harian=("rata_harian", "mean"),
                   persen_hari_aktif=("persen_hari_aktif", "mean"),
                   volatilitas=("volatilitas", "mean"),
                   hari_aktif=("hari_aktif", "mean"),
                   total_diterima=("TOTAL", "sum"))
              .round(3))
    profil["porsi_total_diterima_%"] = (profil["total_diterima"] / aktif["TOTAL"].sum() * 100).round(1)
    st.dataframe(profil, width='stretch')

    st.subheader("Status penyelesaian per klaster")
    st.dataframe(pd.crosstab(aktif["kelompok"], aktif["Status"]), width='stretch')

    st.subheader("Silhouette per klaster")
    st.dataframe(aktif.groupby("kelompok")["silhouette"].mean().round(3).rename("Rata-rata silhouette").to_frame(),
                 width='stretch')

    st.subheader("Uji kestabilan klaster")
    if st.button("Jalankan uji kestabilan"):
        ari, wcss = [], []
        for s in range(n_seed_uji):
            ks = KMeans(n_clusters=k_final, init="k-means++", n_init=int(n_init), random_state=s).fit(Z)
            ari.append(adjusted_rand_score(km.labels_, ks.labels_))
            wcss.append(ks.inertia_)
        st.write(f"ARI min = **{min(ari):.3f}**, rata-rata = **{np.mean(ari):.3f}** "
                 f"· WCSS min-maks = {min(wcss):.3f} - {max(wcss):.3f} ({n_seed_uji} seed)")

# ---- tab 4
with tab4:
    st.subheader("Hasil klasterisasi per petugas")
    kolom_out = ["Nama Petugas", "TOTAL", "Target", "Status"] + ATRIBUT + ["hari_aktif"]
    hasil = pd.concat([
        aktif[kolom_out + ["kelompok"]],
        tidak_aktif[kolom_out].assign(kelompok="Tanpa laporan"),
    ], ignore_index=True)

    filter_kel = st.multiselect("Filter kelompok", sorted(hasil["kelompok"].unique()),
                                default=sorted(hasil["kelompok"].unique()))
    cari = st.text_input("Cari nama petugas")
    tampil = hasil[hasil["kelompok"].isin(filter_kel)]
    if cari:
        tampil = tampil[tampil["Nama Petugas"].str.contains(cari, case=False, na=False)]
    st.dataframe(tampil.round(3), width='stretch', hide_index=True)
    st.caption(f"{len(tampil)} dari {len(hasil)} petugas")

    st.download_button("Unduh hasil (CSV)", hasil.to_csv(index=False).encode("utf-8-sig"),
                       "hasil_klasterisasi_ppl.csv", "text/csv")

# ---- tab 5
with tab5:
    st.subheader("Pemeriksaan kualitas data")
    st.write(f"Berkas memuat **{cek['n_petugas']}** petugas dan **{cek['n_hari']}** hari ({cek['rentang']}).")
    st.write(f"Nilai kosong: **{cek['nilai_kosong']}** · Target = 0: **{cek['target_nol']}**")

    st.subheader("Verifikasi sifat kumulatif kolom tanggal")
    st.write(f"TOTAL sama dengan nilai maksimum baris: **{cek['total_sama_maks']}** dari {cek['n_petugas']}")
    st.write(f"Jumlah laporan baru per hari = TOTAL: **{cek['total_cocok']}** dari {cek['n_petugas']}")
    st.write(f"Selisih negatif: **{cek['selisih_negatif']}**")
    if cek["selisih_negatif"] == 0 and cek["total_cocok"] == cek["n_petugas"]:
        st.success("Verifikasi lolos: kolom tanggal bersifat kumulatif dan berhasil diubah menjadi laporan baru per hari.")
    else:
        st.warning("Ada baris yang tidak konsisten dengan asumsi kumulatif. Periksa data sumber.")

    st.markdown(
        """
**Alur pengolahan**
1. Kolom tanggal (dd/mm) diperlakukan sebagai jumlah laporan *kumulatif*, lalu diubah menjadi selisih (laporan baru per hari).
2. Atribut: rasio penyelesaian (maks. sesuai batas), rata-rata harian, persen hari aktif, volatilitas.
3. Petugas tanpa laporan sama sekali dipisahkan dari proses klasterisasi.
4. Transformasi `log1p` pada rata-rata harian dan volatilitas, lalu normalisasi Min-Max.
5. K-Means (k-means++); klaster diurutkan dari rasio penyelesaian tertinggi ke terendah.
"""
    )
