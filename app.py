import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Teaching Assistant Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# -------------------- CONFIG --------------------
DATA_PATH = "Data/tae.csv"
MODEL_PATH = "models/ta_best_model.pkl"
SCALER_PATH = "models/ta_scaler.pkl"

BASE_COLUMNS = ["Native_teacher", "Instructor", "Course", "Semester", "Class_size"]
TARGET_COL = "Class"

# -------------------- LOAD ARTIFACTS --------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

@st.cache_data
def load_training_data():
    df = pd.read_csv(DATA_PATH, header=None)
    df.columns = ["Native_teacher", "Instructor", "Course", "Semester", "Class_size", "Class"]
    return df

model, scaler = load_artifacts()
df_train = load_training_data()

# -------------------- UI STYLING --------------------
st.markdown("""
<style>
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.main-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: white;
    margin-bottom: 0.2rem;
}
.sub-title {
    color: #b8c1d1;
    font-size: 1rem;
    margin-bottom: 1.8rem;
}
.card {
    background: linear-gradient(135deg, #0f172a, #111827);
    padding: 1.2rem;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 8px 24px rgba(0,0,0,0.20);
    margin-bottom: 1rem;
}
.metric-card {
    background: #0b1220;
    padding: 1rem;
    border-radius: 16px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
}
.metric-title {
    color: #93a4bf;
    font-size: 0.9rem;
}
.metric-value {
    color: white;
    font-size: 1.5rem;
    font-weight: 700;
}
.pred-box {
    color: white;
    padding: 1rem 1.2rem;
    border-radius: 16px;
    font-weight: 700;
    font-size: 1.2rem;
    text-align: center;
    margin-top: 1rem;
}
.small-note {
    color: #9aa8bd;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown('<div class="main-title">🎓 Teaching Assistant Performance Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Predict whether teaching assistant performance is <b>Low</b>, <b>Medium</b>, or <b>High</b> using your trained machine learning pipeline.</div>',
    unsafe_allow_html=True
)

# -------------------- FEATURE ENGINEERING --------------------
def safe_std(series):
    value = series.std()
    if pd.isna(value):
        return 0.0
    return float(value)

def build_training_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df_fe = df.copy()

    # labels for charts / possible training compatibility
    df_fe["Class_label"] = df_fe["Class"].map({1: "Low", 2: "Medium", 3: "High"})

    # binary style features
    df_fe["is_native"] = (df_fe["Native_teacher"] == 1).astype(int)
    df_fe["is_summer"] = (df_fe["Semester"] == 1).astype(int)

    # size-based features
    df_fe["size_log"] = np.log1p(df_fe["Class_size"])
    df_fe["size_squared"] = df_fe["Class_size"] ** 2

    # binned size
    try:
        df_fe["size_binned"] = pd.qcut(
            df_fe["Class_size"],
            q=4,
            labels=False,
            duplicates="drop"
        ).astype(float)
    except Exception:
        df_fe["size_binned"] = 0.0

    # instructor aggregates
    instr_size_mean = df_fe.groupby("Instructor")["Class_size"].mean()
    instr_class_mean = df_fe.groupby("Instructor")["Class"].mean()
    instr_class_std = df_fe.groupby("Instructor")["Class"].apply(safe_std)
    instr_high_rate = df_fe.groupby("Instructor")["Class"].apply(lambda x: (x == 3).mean())

    df_fe["instructor_avg_size"] = df_fe["Instructor"].map(instr_size_mean)
    df_fe["instr_class_mean"] = df_fe["Instructor"].map(instr_class_mean)
    df_fe["instr_class_std"] = df_fe["Instructor"].map(instr_class_std)
    df_fe["instr_high_rate"] = df_fe["Instructor"].map(instr_high_rate)

    # course aggregates
    course_size_mean = df_fe.groupby("Course")["Class_size"].mean()
    course_class_mean = df_fe.groupby("Course")["Class"].mean()
    course_class_std = df_fe.groupby("Course")["Class"].apply(safe_std)
    course_high_rate = df_fe.groupby("Course")["Class"].apply(lambda x: (x == 3).mean())

    df_fe["course_avg_size"] = df_fe["Course"].map(course_size_mean)
    df_fe["course_class_mean"] = df_fe["Course"].map(course_class_mean)
    df_fe["course_class_std"] = df_fe["Course"].map(course_class_std)
    df_fe["course_high_rate"] = df_fe["Course"].map(course_high_rate)

    # interaction features
    df_fe["size_vs_instructor_avg"] = df_fe["Class_size"] - df_fe["instructor_avg_size"]
    df_fe["size_vs_course_avg"] = df_fe["Class_size"] - df_fe["course_avg_size"]
    df_fe["native_x_summer"] = df_fe["is_native"] * df_fe["is_summer"]
    df_fe["native_x_class_size"] = df_fe["is_native"] * df_fe["Class_size"]
    df_fe["summer_x_class_size"] = df_fe["is_summer"] * df_fe["Class_size"]

    return df_fe

def build_single_input_features(input_df: pd.DataFrame, df_train_raw: pd.DataFrame) -> pd.DataFrame:
    df_train_fe = build_training_feature_frame(df_train_raw)

    row = input_df.copy()

    # global fallbacks
    global_class_mean = float(df_train_raw["Class"].mean())
    global_class_std = safe_std(df_train_raw["Class"])
    global_high_rate = float((df_train_raw["Class"] == 3).mean())
    global_size_mean = float(df_train_raw["Class_size"].mean())

    # base engineered fields
    row["is_native"] = (row["Native_teacher"] == 1).astype(int)
    row["is_summer"] = (row["Semester"] == 1).astype(int)
    row["size_log"] = np.log1p(row["Class_size"])
    row["size_squared"] = row["Class_size"] ** 2

    # same qcut bins using training distribution
    try:
        _, bins = pd.qcut(
            df_train_raw["Class_size"],
            q=4,
            retbins=True,
            duplicates="drop"
        )
        row["size_binned"] = pd.cut(
            row["Class_size"],
            bins=bins,
            labels=False,
            include_lowest=True
        ).astype(float)
        row["size_binned"] = row["size_binned"].fillna(0.0)
    except Exception:
        row["size_binned"] = 0.0

    # instructor mappings
    instr_size_mean = df_train_raw.groupby("Instructor")["Class_size"].mean()
    instr_class_mean = df_train_raw.groupby("Instructor")["Class"].mean()
    instr_class_std = df_train_raw.groupby("Instructor")["Class"].apply(safe_std)
    instr_high_rate = df_train_raw.groupby("Instructor")["Class"].apply(lambda x: (x == 3).mean())

    row["instructor_avg_size"] = row["Instructor"].map(instr_size_mean).fillna(global_size_mean)
    row["instr_class_mean"] = row["Instructor"].map(instr_class_mean).fillna(global_class_mean)
    row["instr_class_std"] = row["Instructor"].map(instr_class_std).fillna(global_class_std)
    row["instr_high_rate"] = row["Instructor"].map(instr_high_rate).fillna(global_high_rate)

    # course mappings
    course_size_mean = df_train_raw.groupby("Course")["Class_size"].mean()
    course_class_mean = df_train_raw.groupby("Course")["Class"].mean()
    course_class_std = df_train_raw.groupby("Course")["Class"].apply(safe_std)
    course_high_rate = df_train_raw.groupby("Course")["Class"].apply(lambda x: (x == 3).mean())

    row["course_avg_size"] = row["Course"].map(course_size_mean).fillna(global_size_mean)
    row["course_class_mean"] = row["Course"].map(course_class_mean).fillna(global_class_mean)
    row["course_class_std"] = row["Course"].map(course_class_std).fillna(global_class_std)
    row["course_high_rate"] = row["Course"].map(course_high_rate).fillna(global_high_rate)

    # interactions
    row["size_vs_instructor_avg"] = row["Class_size"] - row["instructor_avg_size"]
    row["size_vs_course_avg"] = row["Class_size"] - row["course_avg_size"]
    row["native_x_summer"] = row["is_native"] * row["is_summer"]
    row["native_x_class_size"] = row["is_native"] * row["Class_size"]
    row["summer_x_class_size"] = row["is_summer"] * row["Class_size"]

    return row

def align_features_for_model(feature_df: pd.DataFrame, model_obj, scaler_obj):
    expected_cols = None

    if hasattr(scaler_obj, "feature_names_in_"):
        expected_cols = list(scaler_obj.feature_names_in_)
    elif hasattr(model_obj, "feature_names_in_"):
        expected_cols = list(model_obj.feature_names_in_)

    if expected_cols is None:
        return feature_df.copy(), [], list(feature_df.columns)

    aligned = feature_df.copy()

    # add missing expected columns
    missing = [c for c in expected_cols if c not in aligned.columns]
    for c in missing:
        aligned[c] = 0.0

    # keep only expected columns in correct order
    aligned = aligned[expected_cols]

    return aligned, missing, expected_cols

# -------------------- INPUT LAYOUT --------------------
left_col, right_col = st.columns([1.15, 0.85], gap="large")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Enter Teaching Details")

    a, b = st.columns(2)

    with a:
        native_teacher = st.selectbox(
            "Native Teacher",
            [1, 2],
            format_func=lambda x: "English Speaker" if x == 1 else "Non-English"
        )
        instructor = st.number_input("Instructor ID", min_value=1, step=1, value=1)
        course = st.number_input("Course ID", min_value=1, step=1, value=1)

    with b:
        semester = st.selectbox(
            "Semester",
            [1, 2],
            format_func=lambda x: "Summer" if x == 1 else "Regular"
        )
        class_size = st.slider("Class Size", min_value=1, max_value=100, value=25)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction")
    predict_btn = st.button("Predict Performance", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Input Summary")

    r1, r2 = st.columns(2)
    with r1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Native Teacher</div><div class="metric-value">{"Yes" if native_teacher == 1 else "No"}</div></div>',
            unsafe_allow_html=True
        )
    with r2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Semester</div><div class="metric-value">{"Summer" if semester == 1 else "Regular"}</div></div>',
            unsafe_allow_html=True
        )

    r3, r4, r5 = st.columns(3)
    with r3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Instructor</div><div class="metric-value">{instructor}</div></div>',
            unsafe_allow_html=True
        )
    with r4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Course</div><div class="metric-value">{course}</div></div>',
            unsafe_allow_html=True
        )
    with r5:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Class Size</div><div class="metric-value">{class_size}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="small-note">This app rebuilds the same feature style used during training and then applies your saved scaler and model for prediction.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- PREDICTION --------------------
if predict_btn:
    input_df = pd.DataFrame([{
        "Native_teacher": native_teacher,
        "Instructor": instructor,
        "Course": course,
        "Semester": semester,
        "Class_size": class_size
    }])

    try:
        engineered_input = build_single_input_features(input_df, df_train)
        aligned_input, missing_added, expected_cols = align_features_for_model(engineered_input, model, scaler)

        scaled_input = scaler.transform(aligned_input)
        prediction = model.predict(scaled_input)[0]

        label_map = {1: "Low", 2: "Medium", 3: "High"}
        color_map = {"Low": "#dc2626", "Medium": "#d97706", "High": "#16a34a"}

        pred_label = label_map.get(int(prediction), str(prediction))
        pred_color = color_map.get(pred_label, "#2563eb")

        st.markdown(
            f'<div class="pred-box" style="background:{pred_color};">Predicted Performance: {pred_label}</div>',
            unsafe_allow_html=True
        )

        if pred_label == "High":
            st.success("This teaching assistant is predicted to perform at a high level.")
        elif pred_label == "Medium":
            st.warning("This teaching assistant is predicted to perform at a medium level.")
        else:
            st.error("This teaching assistant is predicted to perform at a low level.")

        with st.expander("View model input details"):
            st.write("Model input columns used:")
            st.write(expected_cols if expected_cols else list(aligned_input.columns))

            if missing_added:
                st.write("Columns that were missing and auto-filled with 0.0:")
                st.write(missing_added)

            st.write("Final input row passed to the scaler/model:")
            st.dataframe(aligned_input)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -------------------- FOOTER --------------------
st.markdown("<br>", unsafe_allow_html=True)
st.caption("Built with Streamlit • ML Project by Uwais Muhammed")