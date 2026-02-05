import streamlit as st
import pandas as pd
import os
import sys

# Add current directory to path so we can import src modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import data_manager, cleaner, model_engine, visualizer

st.set_page_config(page_title="Antigravity 數據處理模組", layout="wide")

# --- Auth Check ---
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["app_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    # 1. Check if configured
    try:
        if "app_password" not in st.secrets:
            # No password set -> Open Access (or Warning)
            st.sidebar.warning("⚠️ 未設定密碼 (app_password)。網站目前公開。")
            return True
            
    except Exception:
        # Local run without secrets.toml
        # st.sidebar.info("ℹ️ 本地模式 (無 secrets.toml)：略過密碼驗證。")
        return True

    # 2. Check session state
    if "password_correct" not in st.session_state:
        # First run, show input
        st.text_input(
            "請輸入系統密碼 (Password)", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input again
        st.text_input(
            "請輸入系統密碼 (Password)", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 密碼錯誤")
        return False
    else:
        # Password correct
        return True

if not check_password():
    st.stop()

# --- Global Definitions ---
TYPE_MAPPING = {
    "不使用 (Unused)": "Unused",
    "ID (識別碼)": "ID",
    "數值特徵 (Numerical)": "Numerical",
    "類別特徵 (Categorical)": "Categorical",
    "時間特徵 (Datetime)": "Datetime",
    "預測目標 (Target)": "Target"
}
REVERSE_TYPE_MAPPING = {v: k for k, v in TYPE_MAPPING.items()}
OPTIONS_DISPLAY = list(TYPE_MAPPING.keys())

st.title("Antigravity 智能預測平台 (AI Prediction Platform) 🚀")

# --- Sidebar Configuration ---
st.sidebar.header("系統設定 (Configuration)")
st.sidebar.text("Debug: Options loaded: " + str(len(OPTIONS_DISPLAY))) # Debug info
data_source_path = st.sidebar.text_input("資料來源路徑 (Data Source Path)", value=data_manager.DATA_DIR)

st.sidebar.markdown("---")
st.sidebar.header("模型版本管理 (Model Versioning)")
if 'model_engine' not in st.session_state:
    st.session_state.model_engine = model_engine.ModelEngine()

available_models = st.session_state.model_engine.list_models()
if not available_models:
    model_options = ["尚無模型 (No Models)"]
    selected_model_name = model_options[0]
else:
    model_options = available_models
    # Default to first (latest)
    selected_model_name = st.sidebar.selectbox("選擇使用模型版本", model_options)

# Auto-load logic
if 'current_model_name' not in st.session_state:
    st.session_state.current_model_name = None

if selected_model_name != "尚無模型 (No Models)" and selected_model_name != st.session_state.current_model_name:
    # Load the model
    try:
        model_path = os.path.join(model_engine.MODEL_DIR, selected_model_name)
        st.session_state.current_model = st.session_state.model_engine.load_model(model_path)
        st.session_state.current_model_name = selected_model_name
        st.sidebar.success(f"已載入: {selected_model_name}")
    except Exception as e:
        st.sidebar.error(f"載入失敗: {e}")

    except Exception as e:
        st.sidebar.error(f"載入失敗: {e}")

    except Exception as e:
        st.sidebar.error(f"載入失敗: {e}")

# --- Model Import/Export (Manual) ---
st.sidebar.markdown("---")
st.sidebar.header("模型存取 (Import/Export)")

# 1. Download Current Model
if st.session_state.current_model_name:
    local_path = os.path.join(model_engine.MODEL_DIR, st.session_state.current_model_name)
    if os.path.exists(local_path):
        with open(local_path, "rb") as f:
            st.sidebar.download_button(
                label="📥 下載此模型 (.joblib)",
                data=f,
                file_name=st.session_state.current_model_name,
                mime="application/octet-stream"
            )

# 2. Upload External Model
uploaded_model = st.sidebar.file_uploader("📤 上傳舊模型 (Restore)", type=["joblib"], key="model_restore")
if uploaded_model:
    # Save to models directory
    restore_path = os.path.join(model_engine.MODEL_DIR, uploaded_model.name)
    with open(restore_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    
    st.sidebar.success(f"已還原: {uploaded_model.name}")
    
    # Reload functionality
    if st.sidebar.button("載入此模型 (Load Uploaded)"):
        try:
            st.session_state.current_model = st.session_state.model_engine.load_model(restore_path)
            st.session_state.current_model_name = uploaded_model.name
            st.rerun()
        except:
             st.sidebar.error("載入失敗，檔案可能損毀")

st.sidebar.markdown("---")
st.sidebar.header("選擇工作流程 (Workflow)")
workflow = st.sidebar.selectbox(
    "請選擇 AI 任務類型",
    [
        "1. 🎯 下單機率預測 (Purchase Prediction)",
        "2. 👥 客群分群分析 (Segmentation)",
        "3. 💰 消費金額預測 (Value Prediction)",
        "4. 🕵️‍♂️ 潛在客戶挖掘 (PU Learning)"
    ]
)

# Initialize Session State
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
# PU Learning needs two dataframes
if 'df_pos' not in st.session_state:
    st.session_state.df_pos = None
if 'df_unlabeled' not in st.session_state:
    st.session_state.df_unlabeled = None

if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = {}
if 'model_engine' not in st.session_state:
    st.session_state.model_engine = model_engine.ModelEngine()
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

# --- Main Interface ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. 資料準備 (Data Prep)", 
    "2. 模型訓練 (Model Training)", 
    "3. 預測與應用 (Prediction)",
    "4. 修正與優化 (Feedback)",
    "5. 戰情儀表板 (Dashboard)"
])

# === TAB 1: DATA PREP ===
with tab1:
    st.header("資料匯入與欄位定義")
    
    if "PU Learning" in workflow:
        st.info("🕵️‍♂️ PU Learning 需要兩份資料：已被標記的正向名單 (File A) 與 未標記名單 (File B)")
        col_u1, col_u2 = st.columns(2)
        
        with col_u1:
            up_pos = st.file_uploader("上傳正向名單 (File A - 已購客)", type=["csv"], key="pu_pos")
            if up_pos:
                st.session_state.df_pos = data_manager.load_csv_robust(up_pos)
                st.success(f"已載入正向樣本: {len(st.session_state.df_pos)} 筆")
                
        with col_u2:
            up_un = st.file_uploader("上傳未標記名單 (File B - 潛在客)", type=["csv"], key="pu_un")
            if up_un:
                st.session_state.df_unlabeled = data_manager.load_csv_robust(up_un)
                st.success(f"已載入未標記樣本: {len(st.session_state.df_unlabeled)} 筆")
        
        # Combine for metadata definition (taking mostly from B as it's the target space)
        # But we need to make sure columns match.
        if st.session_state.df_pos is not None and st.session_state.df_unlabeled is not None:
             # Concatenate for metadata view
             st.session_state.df_raw = pd.concat([st.session_state.df_pos, st.session_state.df_unlabeled], ignore_index=True)
    else:
        # Standard Single File Upload
        uploaded_file = st.file_uploader("上傳訓練資料 (CSV)", type=["csv"], key="train_uploader")
        if uploaded_file is not None:
            try:
                st.session_state.df_raw = data_manager.load_csv_robust(uploaded_file)
                st.success(f"成功載入 {uploaded_file.name}，資料形狀: {st.session_state.df_raw.shape}")
            except Exception as e:
                st.error(f"檔案讀取失敗: {e}")

    # Metadata Mapping Section
    if st.session_state.df_raw is not None:
        st.subheader("欄位屬性定義")
        df = st.session_state.df_raw
        cols = df.columns.tolist()
        
        # Determine target logic defaults
        # If clustering, hide/disable Target option? Or just let user ignore it.
        # Let's keep it consistent but guide user with text.
        if "Segmentation" in workflow:
            st.info("ℹ️ 分群分析為非監督式學習，不需要設定『預測目標 (Target)』。")
        
        col1, col2 = st.columns(2)
        with col1:
            updated_metadata = {}
            for col in cols:
                current_backend_type = st.session_state.metadata.get(col, "Unused")
                default_display = REVERSE_TYPE_MAPPING.get(current_backend_type, "不使用 (Unused)")
                
                try:
                    default_idx = OPTIONS_DISPLAY.index(default_display)
                except:
                    default_idx = 0
                
                selected_display = st.selectbox(f"'{col}' 的屬性", OPTIONS_DISPLAY, index=default_idx, key=f"sel_{col}")
                backend_type = TYPE_MAPPING[selected_display]
                
                if backend_type != "Unused":
                    updated_metadata[col] = backend_type
                    
        with col2:
            st.json(updated_metadata)
            if st.button("儲存欄位設定 (Save Metadata)"):
                st.session_state.metadata = updated_metadata
                data_manager.save_metadata(updated_metadata)
                st.success("欄位設定已儲存！")

        # Processing Section
        st.subheader("執行資料清洗")
        if st.button("按照設定執行清洗 (Run Cleaning)"):
             if not st.session_state.metadata:
                st.error("請先定義並儲存欄位設定！")
             else:
                with st.spinner("資料清洗與特徵工程中..."):
                    options = {"batch_id": "manual_run"} # Default options for now, can expand later
                    try:
                        df_proc = cleaner.clean_data(st.session_state.df_raw, st.session_state.metadata, options)
                        st.session_state.df_processed = df_proc
                        data_manager.save_processed_data(df_proc)
                        st.success("清洗完成！可前往『模型訓練』分頁。")
                        st.dataframe(df_proc.head())
                    except Exception as e:
                        st.error(f"處理失敗: {e}")

# === TAB 2: MODEL TRAINING ===
with tab2:
    st.header(f"模型訓練: {workflow}")
    
    if st.session_state.df_processed is None:
        st.warning("請先在第一頁完成資料準備。")
    else:
        df_train = st.session_state.df_processed
        
        # Identify special columns
        target_col = next((c for c, t in st.session_state.metadata.items() if t == 'Target'), None)
        id_col = next((c for c, t in st.session_state.metadata.items() if t == 'ID'), None)
        
        # UI Logic based on Workflow
        if "Purchase Prediction" in workflow:
            is_lookalike = st.checkbox("僅包含正向樣本 (純下單名單 / Lookalike Modeling)", value=True, help="如果您的訓練檔案只有「已下單」的客戶，請勾選此項。系統將尋找與這群人相似的潛在客戶。")
            
            if is_lookalike:
                st.info("模式：相似受眾分析 (Lookalike)。系統將學習此名單的特徵分佈，找出類似的潛在客戶。")
                if st.button("開始訓練 (Train Lookalike Model)"):
                    with st.spinner("訓練潛在受眾模型中..."):
                        engine = st.session_state.model_engine
                        # Lookalike doesn't use target column
                        model, metrics = engine.train_lookalike(df_train, id_col)
                        st.session_state.current_model = model
                        
                        st.success(f"訓練完成！已學習 {metrics['num_samples']} 筆正向樣本。")
                        
                         # Feature Importance (IsolationForest implies importance via split features, but sklearn doesn't provide it easily. Skiping plot or using simple variance?)
                        st.write("模型已準備好進行名單預測。請前往第三頁。")
                        
                        # Save
                        path, name = engine.save_model(model, "lookalike")
                        st.info(f"模型已儲存: {name}")

            else:
                # Standard Classification
                if not target_col:
                    st.error("標準分類模式需要定義『預測目標 (Target)』欄位！請回第一頁設定，或勾選上方『僅包含正向樣本』。")
                else:
                    st.write(f"預測目標: **{target_col}** (分類任務)")
                    if st.button("開始訓練 (Train Classifier)"):
                        with st.spinner("訓練中..."):
                            engine = st.session_state.model_engine
                            model, metrics = engine.train_classification(df_train, target_col, id_col)
                            st.session_state.current_model = model
                            
                            st.success("訓練完成！")
                            visualizer.plot_classification_metrics(metrics)
                            
                            # Feature Importance
                            feature_cols = [c for c in df_train.columns if c not in [target_col, id_col, 'Batch_ID']]
                            st.subheader("特徵重要性分析 (Feature Analysis)")
                            visualizer.plot_feature_importance(model, feature_cols)
                            
                            # Textual Explanation
                            insights = visualizer.explain_feature_importance(model, feature_cols)
                            for line in insights:
                                st.markdown(line)
                            
                            # Save
                            path, name = engine.save_model(model, "classifier")
                            st.info(f"模型已儲存: {name}")

        elif "Segmentation" in workflow:
            st.write("分群分析 (Clustering)")
            k_clusters = st.slider("預計分群數量 (K)", 2, 10, 3, help="如果不知道選多少，系統會自動嘗試尋找最佳值")
            auto_k = st.checkbox("自動尋找最佳 K 值 (Auto K)", value=True)
            
            if st.button("執行分群 (Run Clustering)"):
                with st.spinner("分群運算中..."):
                    engine = st.session_state.model_engine
                    k_arg = None if auto_k else k_clusters
                    model, metrics, labels = engine.train_clustering(df_train, id_col=id_col, k=k_arg)
                    st.session_state.current_model = model
                    
                    st.success(f"分群完成！最佳群數: {metrics['k']} (Silhouette: {metrics['silhouette_score']:.3f})")
                    
                    # Plotting
                    df_viz = df_train.copy()
                    df_viz['Cluster'] = labels
                    valid_cols = [c for c in df_viz.columns if pd.api.types.is_numeric_dtype(df_viz[c]) and c not in ['Cluster', 'Batch_ID']]
                    
                    visualizer.plot_clusters_2d(df_viz, 'Cluster', valid_cols)

        elif "PU Learning" in workflow:
            st.write("🕵️‍♂️ 潛在客戶挖掘 (Positive-Unlabeled Learning)")
            
            if st.session_state.df_pos is None or st.session_state.df_unlabeled is None:
                 st.error("請先在第一頁上傳 File A (正向) 與 File B (未標記)！")
            else:
                 # Manual Hints
                 st.info("此模型會自動區分正向與未標記資料特徵。您也可以手動加強某些關鍵特徵的權重。")
                 
                 # Feature Weight Config
                 with st.expander("⚙️ 進階設定：特徵加權 (Feature Weights)"):
                     st.write("設定權重 (預設 1.0)。設為 0 代表該特徵不參與加權調整。")
                     feature_cols = [c for c in st.session_state.df_unlabeled.columns if c not in [id_col, 'Batch_ID']]
                     
                     weights = {}
                     cols = st.columns(3)
                     for i, col in enumerate(feature_cols):
                         with cols[i % 3]:
                             # Default 1.0
                             val = st.number_input(f"{col}", 0.0, 5.0, 1.0, 0.1, key=f"w_{col}")
                             if val != 1.0:
                                 weights[col] = val
                 
                 if st.button("開始挖掘 (Train PU Model)"):
                     with st.spinner("正在進行 PU Learning 訓練..."):
                         engine = st.session_state.model_engine
                         
                         # Need to pass separated DF A and B for cleaning?
                         # Usually we clean merged DF then split.
                         # Our Tab 1 merged them into df_raw and ran cleaner -> df_processed.
                         # Now we need to split df_processed back into Pos and Unlabeled based on index or source?
                         # Tricky.
                         # Easier approach: Clean df_pos and df_unlabeled SEPARATELY using same metadata options?
                         # Or just split df_processed.
                         
                         # Since df_raw was concat(pos, unlabeled), the first len(pos) rows are pos.
                         n_pos = len(st.session_state.df_pos)
                         df_proc = st.session_state.df_processed
                         
                         df_train_pos = df_proc.iloc[:n_pos].copy()
                         df_train_un = df_proc.iloc[n_pos:].copy()
                         
                         # Train
                         model, metrics = engine.train_pu_learning(df_train_pos, df_train_un, weights, id_col)
                         st.session_state.current_model = model
                         
                         st.success(f"訓練完成！AUC: {metrics['auc']:.4f}")
                         st.write(f"使用正樣本數: {metrics['pos_samples']}, 負樣本數(採樣): {metrics['neg_samples_used']}")
                         
                         # Feature Importance (if pipeline)
                         # Extract from pipeline step 'clf' coefficients
                         # PU module handles this internally? 
                         # Let's try to extract coefficient info if available
                         try:
                             # Access inner pipeline
                             # model is CalibratedCV in app?
                             # engine returns (model, metrics)
                             # Wait, engine returns (calibrated_clf, metrics)
                             # We need the base estimator to get coefs.
                             # CalibratedClassifierCV -> calibrated_classifiers_[0].estimator (if prefit) or base_estimator
                             
                             # Actually engine.train_pu_learning returns (calibrated_clf, metrics)
                             # Getting feature importance from calibrated SVM/Logistic is hard visually.
                             pass
                         except:
                             pass
                         
                         # Save
                         path, name = engine.save_model(model, "pu_model")
                         st.info(f"模型已儲存: {name}")

        elif "Value Prediction" in workflow:
            if not target_col:
                st.error("此模式需要定義『預測目標 (Target)』欄位！")
            else:
                st.write(f"預測目標: **{target_col}** (回歸任務)")
                if st.button("開始訓練 (Train Regressor)"):
                    with st.spinner("訓練中..."):
                        engine = st.session_state.model_engine
                        model, metrics = engine.train_regression(df_train, target_col, id_col)
                        st.session_state.current_model = model
                        
                        st.success(f"訓練完成! MSE: {metrics['mse']:.4f}, R2: {metrics['r2']:.4f}")
                        
                        feature_cols = [c for c in df_train.columns if c not in [target_col, id_col, 'Batch_ID']]
                        st.subheader("特徵重要性分析 (Feature Analysis)")
                        visualizer.plot_feature_importance(model, feature_cols)
                        
                        # Textual Explanation
                        insights = visualizer.explain_feature_importance(model, feature_cols)
                        for line in insights:
                            st.markdown(line)

# === TAB 3: PREDICTION (Mode B) ===
with tab3:
    st.header("應用與預測 (File B)")
    
    if st.session_state.current_model is None:
        st.warning("請先訓練模型或載入已有模型。")
    else:
        st.info("使用當前訓練好的模型進行預測。")
        upload_pred = st.file_uploader("上傳預測名單 (File B)", type=["csv"], key="pred_uploader")
        
        if upload_pred:
            df_pred_raw = data_manager.load_csv_robust(upload_pred)
            
            if st.button("執行預測 (Generate Predictions)"):
                # Ideally we need to run same cleaning/pipeline on prediction data.
                # For Phase 2 MVP, we assume user uploads pre-processed or we reuse clean options
                # This is a critical TODO: persist cleaning pipeline.
                # For now, let's just run simple cleaning using current metadata
                options = {"batch_id": "prediction_run"}
                try:
                    df_pred_clean = cleaner.clean_data(df_pred_raw, st.session_state.metadata, options)
                    
                    # Ensure we pass the ID column so it gets dropped before prediction
                    id_col = next((c for c, t in st.session_state.metadata.items() if t == 'ID'), None)
                    
                    engine = st.session_state.model_engine
                    preds, probs = engine.predict(st.session_state.current_model, df_pred_clean, id_col=id_col)
                    
                    df_result = df_pred_raw.copy() # Attach to original?
                    df_result['Predicted_Result'] = preds
                    if probs is not None:
                        df_result['Probability_Score'] = probs
                        
                    st.success("預測完成！")
                    st.dataframe(df_result.head())
                    
                    # Download
                    csv = df_result.to_csv(index=False).encode('utf-8-sig') # ensuring utf-8-sig for excel
                    st.download_button(
                        "下載預測結果 (Download CSV)",
                        csv,
                        "prediction_results.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
                except Exception as e:
                    st.error(f"預測流程錯誤: {e}")
                    st.warning("提示: 預測名單的欄位結構必須與訓練資料一致。如果遇到 Feature mismatch，請檢查『編號』欄位是否已在第一頁正確設定為 'ID' (ID 屬性不會被用作特徵)。")


# === TAB 4: FEEDBACK (Mode C) ===
with tab4:
    st.header("修正與優化 (Feedback Loop)")
    st.info("此步驟用於將「實際執行結果 (File C)」回饋給 AI，以持續優化模型準確度。")
    
    if st.session_state.current_model is None:
        st.warning("請先有訓練好的模型 (v1.0)，才能進行優化 (v1.1)。")
    else:
        upload_feedback = st.file_uploader("上傳實際結果資料 (File C)", type=["csv"], key="feedback_uploader")
        
        if upload_feedback:
             try:
                 df_feedback_raw = data_manager.load_csv_robust(upload_feedback)
                 st.write("回饋資料預覽:", df_feedback_raw.head())
                 
                 if st.button("執行模型優化 (Update Model)"):
                    with st.spinner("正在根據回饋資料重新調校模型..."):
                        try:
                            # Process Feedback Data
                            options = {"batch_id": "feedback_run"}
                            df_feedback_clean = cleaner.clean_data(df_feedback_raw, st.session_state.metadata, options)
                            
                            id_col = next((c for c, t in st.session_state.metadata.items() if t == 'ID'), None)
                            target_col = next((c for c, t in st.session_state.metadata.items() if t == 'Target'), None)
                            
                            engine = st.session_state.model_engine
                            current_model = st.session_state.current_model
                            
                            # Determine type
                            if "PU Learning" in workflow:
                                model_type = "pu_learning"
                            elif isinstance(current_model, model_engine.IsolationForest):
                                model_type = "lookalike"
                            else:
                                model_type = "standard"
                            
                            new_model, metrics = engine.update_model(current_model, df_feedback_clean, target_col, id_col, model_type)
                            st.session_state.current_model = new_model
                            
                            st.success("模型優化完成！版本已更新。")
                            if "mse" in metrics:
                                 st.write(f"新模型誤差 (MSE): {metrics['mse']:.4f}")
                            elif "silhouette_score" in metrics:
                                 st.write(f"新分群分數: {metrics['silhouette_score']:.3f}")
                            elif "num_samples" in metrics:
                                 st.write(f"Lookalike 模型已擴充，目前學習樣本數: {metrics['num_samples']}")
                            else:
                                 visualizer.plot_classification_metrics(metrics)
                                 
                            # Save v1.1
                            path, name = engine.save_model(new_model, "model_v1.1")
                            st.info(f"優化後模型已儲存: {name}")
                            st.balloons()
                            
                        except NotImplementedError:
                            st.warning("⚠️ PU Learning 屬於高階模型，建議您將新的購買名單合併至『正向名單 (File A)』後，回到第一頁重新上傳並重新訓練，以獲得最佳效果。")
                        except Exception as e:
                            st.error(f"優化失敗: {e}")
                            st.exception(e)
             except Exception as load_e:
                 st.error(f"讀取失敗: {load_e}")


# === TAB 5: DASHBOARD ===
with tab5:
    st.header("數據戰情儀表板 (Analytics Dashboard)")
    
    # KPIs
    st.subheader("關鍵指標 (KPIs)")
    k1, k2, k3 = st.columns(3)
    
    # Calculating stats (Simplified for MVP - normally would query a DB or log file)
    # We use session state or just count current model properties
    total_models = len(available_models) if available_models else 0
    current_ver = selected_model_name if selected_model_name else "N/A"
    
    k1.metric("可用模型版本數", total_models)
    k2.metric("當前使用版本", current_ver.split('_v')[1].split('.')[0] if '_v' in current_ver else "N/A")
    k3.metric("系統狀態", "🟢 Online")
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("轉換漏斗範例 (Conversion Funnel)")
        # In a real app, we would track: Rows Uploaded -> Rows Predictied -> High Prob Rows -> Actual Orders
        # Here we mock it based on typical workflow or current session data
        current_rows = len(st.session_state.df_raw) if st.session_state.df_raw is not None else 0
        current_preds = 0 # Track if we predicted in session?
        
        # Mock Data for Visualization Demo
        funnel_data = {
            "1. 潛在名單 (File B)": 1000,
            "2. 有效資料 (Valid)": 950,
            "3. 預測高潛力 (High Prob)": 320,
            "4. 實際轉換 (Conversion)": 85
        }
        visualizer.plot_funnel(funnel_data)
        st.caption("*此圖表目前為範例數據，未來將串接實際歷史紀錄")

    with col_b:
        st.subheader("模型準確率趨勢 (Accuracy Trend)")
        # Mock Data: showing improvement over versions
        trend_data = pd.DataFrame([
            {"Version": "v1.0 (2/1)", "Metric": "F1-Score", "Score": 0.65},
            {"Version": "v1.0 (2/1)", "Metric": "Precision", "Score": 0.60},
            {"Version": "v1.1 (2/3)", "Metric": "F1-Score", "Score": 0.72},
            {"Version": "v1.1 (2/3)", "Metric": "Precision", "Score": 0.75},
            {"Version": "v1.2 (Today)", "Metric": "F1-Score", "Score": 0.78},
            {"Version": "v1.2 (Today)", "Metric": "Precision", "Score": 0.82},
        ])
        visualizer.plot_accuracy_trend(trend_data)
        st.caption("*此圖表為模擬趨勢，展示模型經由 Feedback Loop 優化後的成長軌跡")
