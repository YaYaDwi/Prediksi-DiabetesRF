import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE


# Mengatur gaya warna pada seaborn
sns.set_style("whitegrid")
st.set_page_config(page_title="Prediksi Diabetes", page_icon="🩺", layout="centered")

# Fungsi untuk melatih model dan mempersiapkan pipeline
@st.cache_resource
def load_model():
    # Load dataset
    data = pd.read_csv('diabetes.csv')
    
    # Memisahkan fitur dan target
    X = data[['Pregnancies', 'Glucose', 'Insulin', 'DiabetesPedigreeFunction', 'Age']]
    y = data['Outcome']
    
    # Melakukan oversampling dengan SMOTE
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    
    # Normalisasi fitur menggunakan MinMaxScaler
    scaler = MinMaxScaler()
    X_smote_scaled = scaler.fit_transform(X_smote)
    
    # Seleksi fitur menggunakan Chi-Square
    chi2_selector = SelectKBest(score_func=chi2, k=5)
    X_smote_selected = chi2_selector.fit_transform(X_smote_scaled, y_smote)
    
    # Membagi data menjadi data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(X_smote_selected, y_smote, test_size=0.2, random_state=42)
    
    # Membuat model Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluasi model pada data uji
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification = classification_report(y_test, y_pred, zero_division=1)
    
    return rf_model, scaler, chi2_selector, accuracy, confusion, classification, data

# Load model dan pipeline preprocessing
rf_model, scaler, chi2_selector, accuracy, confusion, classification, data = load_model()

# Set up navbar
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Menu", ["Prediksi Diabetes", "Tentang Fitur", "Visualisasi Data", "Akurasi Model"])

# Warna tema untuk tampilan
primary_color = "#2C3E50"  # Warna utama
secondary_color = "#18BC9C"  # Warna sekunder

# Tampilan utama untuk prediksi diabetes
if page == "Prediksi Diabetes":
    st.title("💉 Prediksi Diabetes Menggunakan Random Forest")
    st.write("Masukkan nilai-nilai berikut untuk memprediksi diabetes:")
    
    # Input untuk fitur-fitur yang dipilih, dengan deskripsi yang lebih jelas
    pregnancies = st.number_input(
        "Berapa kali Anda pernah hamil? (Jika Anda pria, masukkan 0)", 
        min_value=0
    )
    # Input kadar glukosa sebagai daftar pilihan
    glucose_option = st.selectbox(
        "Berapa kadar gula darah Anda setelah puasa?",
        [
            "Rendah (< 70 mg/dL)",
            "Normal (70 - 140 mg/dL)",
            "Pra-diabetes (140 - 199 mg/dL)",
            "Diabetes (> 200 mg/dL)"
        ]
    )

    # Konversi pilihan ke nilai numerik untuk model
    glucose_mapping = {
        "Rendah (< 70 mg/dL)": 65,
        "Normal (70 - 140 mg/dL)": 100,
        "Pra-diabetes (140 - 199 mg/dL)": 170,
        "Diabetes (> 200 mg/dL)": 220
    }
    glucose = glucose_mapping[glucose_option]

    # Input kadar insulin sebagai daftar pilihan
    insulin_option = st.selectbox(
        "Berapa kadar insulin Anda?",
        [
            "Rendah (< 15 μU/mL)",
            "Normal (15 - 276 μU/mL)",
            "Tinggi (> 276 μU/mL)"
        ]
    )

    # Konversi pilihan ke nilai numerik untuk model
    insulin_mapping = {
        "Rendah (< 15 μU/mL)": 10,
        "Normal (15 - 276 μU/mL)": 150,
        "Tinggi (> 276 μU/mL)": 300
    }
    insulin = insulin_mapping[insulin_option]

    # Input DPF sebagai daftar pilihan
    dpf_option = st.selectbox(
        "Riwayat Diabetes dalam Keluarga:",
        [
            "Tidak ada riwayat (0.0)",
            "Riwayat keluarga jauh (0.1 - 0.3)",
            "Riwayat keluarga dekat (0.4 - 0.6)",
            "Riwayat keluarga signifikan (0.7 - 1.0)",
            "Risiko keluarga sangat tinggi (>1.0)"
        ]
    )

    # Konversi pilihan ke nilai numerik untuk model
    dpf_mapping = {
        "Tidak ada riwayat (0.0)": 0.0,
        "Riwayat keluarga jauh (0.1 - 0.3)": 0.2,
        "Riwayat keluarga dekat (0.4 - 0.6)": 0.5,
        "Riwayat keluarga signifikan (0.7 - 1.0)": 0.8,
        "Risiko keluarga sangat tinggi (>1.0)": 1.2
    }

    dpf = dpf_mapping[dpf_option]

    age = st.number_input(
        "Berapa usia Anda saat ini? (dalam tahun)", 
        min_value=0
    )


    
    # Tombol untuk prediksi
    if st.button("Prediksi"):
        data_input = np.array([[pregnancies, glucose, insulin, dpf, age]])
        data_input_scaled = scaler.transform(data_input)
        data_input_selected = chi2_selector.transform(data_input_scaled)

        # Prediksi
        prediction = rf_model.predict(data_input_selected)

        # Tampilkan hasil prediksi dengan rekomendasi 0 = tidak terkena 1 = terkena
        if prediction[0] == 1:
            st.error("Hasil Prediksi: Anda kemungkinan terkena diabetes.")
            st.write("### Rekomendasi:")
            st.write("**Makanan:** Kurangi gula dan karbohidrat, konsumsi sayuran hijau, ikan berlemak, kacang-kacangan, dan biji-bijian.")
            st.write("**Aktivitas:** Rutin berolahraga seperti jalan kaki, jogging, atau berenang. Pastikan untuk menjaga berat badan yang sehat.")
        else:
            st.success("Hasil Prediksi: Anda tidak terkena diabetes.")
            st.write("### Rekomendasi:")
            st.write("**Makanan:** Konsumsi makanan seimbang, kaya serat, dan rendah lemak. Tetap batasi konsumsi gula.")
            st.write("**Aktivitas:** Lakukan olahraga ringan hingga sedang seperti bersepeda atau yoga untuk menjaga kesehatan tubuh.")

# Tentang fitur
elif page == "Tentang Fitur":
    st.title("ℹ️ Tentang Fitur")
    st.write("""
    Penjelasan lebih rinci mengenai setiap fitur yang digunakan dalam prediksi:
    
    1. **Pregnancies**: 
       - Jumlah kehamilan yang dialami seorang wanita.
       - Data ini penting karena kehamilan meningkatkan risiko diabetes gestasional, yang dapat memengaruhi hasil diagnosis diabetes.
    
    2. **Glucose**: 
       - Tingkat glukosa dalam darah setelah puasa (mg/dL).
       - Kadar glukosa yang tinggi setelah puasa menunjukkan kemungkinan diabetes tipe 2. 
       - Patokan normal: Glukosa < 140 mg/dL setelah tes toleransi glukosa oral.

    3. **Insulin**: 
       - Kadar insulin dalam darah (μU/mL).
       - Insulin digunakan untuk mengatur kadar glukosa, dan kekurangan insulin bisa menjadi indikator diabetes.
       - Nilai normal: 15-276 μU/mL.

    4. **Diabetes Pedigree Function (DPF)**:
       - Mencerminkan risiko diabetes berdasarkan riwayat keluarga.
       - Nilai DPF yang lebih tinggi mengindikasikan riwayat keluarga dengan diabetes yang signifikan.

    5. **Age**: 
       - Usia pasien.
       - Risiko diabetes meningkat seiring bertambahnya usia, terutama di atas 45 tahun.
    """)

# Visualisasi data
elif page == "Visualisasi Data":
    st.title('📊 Visualisasi Data dan Evaluasi Model')

    # Visualisasi distribusi kelas Outcome
    st.subheader("Distribusi Kelas Outcome")
    st.write("""
    **Outcome**: 
    - 0: Tidak terkena diabetes.
    - 1: Terkena diabetes.
    """)
    fig, ax = plt.subplots()
    sns.countplot(x="Outcome", data=data, palette=[secondary_color, primary_color], ax=ax)
    ax.set_xlabel("Outcome", color=primary_color)
    ax.set_ylabel("Jumlah", color=primary_color)
    ax.set_xticklabels(["Tidak terkena diabetes (0)", "Terkena diabetes (1)"], color=primary_color)
    st.pyplot(fig)

    # Visualisasi distribusi setiap fitur dengan patokan angka
    st.subheader("Distribusi Fitur Berdasarkan Outcome dengan Patokan Angka")
    feature_details = {
        "Pregnancies": "Jumlah kehamilan yang pernah dialami.",
        "Glucose": "Kadar glukosa darah puasa (mg/dL). **Normal < 140 mg/dL.**",
        "Insulin": "Kadar insulin dalam darah (μU/mL). **Normal 15-276 μU/mL.**",
        "DiabetesPedigreeFunction": "Fungsi keturunan diabetes. **Lebih tinggi berarti risiko lebih besar.**",
        "Age": "Usia pasien. **Risiko lebih tinggi untuk usia ≥ 45 tahun.**"
    }
    for feature, detail in feature_details.items():
        st.write(f"**{feature}**: {detail}")
        fig, ax = plt.subplots()
        sns.histplot(data=data, x=feature, hue="Outcome", kde=True, palette=[secondary_color, primary_color], ax=ax)
        ax.set_xlabel(feature, color=primary_color)
        ax.set_ylabel("Jumlah", color=primary_color)
        ax.legend(["Tidak terkena diabetes (0)", "Terkena diabetes (1)"], title="Outcome", loc='upper right')
        st.pyplot(fig)


# Akurasi model
elif page == "Akurasi Model":
    st.title("📈 Evaluasi Model")
    st.write(f"<h4 style='color: {secondary_color};'>Akurasi Model: {accuracy:.2f}</h4>", unsafe_allow_html=True)
    st.write(f"Akurasi model setelah penerapan SMOTE dan seleksi fitur dengan chi-square adalah {accuracy:.2f}.")
    st.write("Confusion Matrix:")
    st.write(confusion)
    st.write("Classification Report:")
    st.text(classification)