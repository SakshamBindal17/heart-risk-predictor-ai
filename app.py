import streamlit as st
import joblib
import numpy as np
import google.generativeai as genai
import os

# --- Configure Gemini API ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# --- Load your Logistic Regression model and scaler ---
model = joblib.load('best_logreg_model.pkl')
scaler = joblib.load('scaler.pkl')

# --- Helper variables ---
feature_names = ['Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol', 'Fasting Blood Sugar',
                 'Rest ECG', 'Max Heart Rate', 'Exercise Angina', 'ST depression', 'Slope', 'Major Vessels', 'Thalassemia']

# --- Streamlit App Start ---
st.title("Heart Disease Risk Predictor with AI Health Assistant")
st.write("Enter your health details. Hover over input labels to get more information.")

# --- User inputs with tooltips ---

age = st.number_input(
    "Age (years)",
    min_value=1,
    max_value=120,
    value=50,
    help="Enter your current age in years.",
    key="age"
)

sex = st.selectbox(
    "Sex",
    options=[0,1],
    format_func=lambda x: "Female" if x == 0 else "Male",
    help="Your biological sex at birth. Select Female if you are a woman, Male if you are a man.",
    key="sex"
)

cp = st.selectbox(
    "Chest Pain Type",
    options=[0,1,2,3],
    format_func=lambda x: {
        0: "Typical angina",
        1: "Atypical angina",
        2: "Non-anginal pain",
        3: "No chest pain"
    }[x],
    help=(
        "Chest Pain Type describes your chest discomfort:\n"
        "- Typical angina: chest pain when active, relieved by rest.\n"
        "- Atypical angina: chest pain not clearly linked to activity.\n"
        "- Non-anginal pain: pain not caused by heart.\n"
        "- No chest pain."
    ),
    key="cp"
)

trestbps = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=80,
    max_value=250,
    value=120,
    help="Pressure in arteries while resting; normal ~120 mm Hg.",
    key="trestbps"
)

chol = st.number_input(
    "Serum Cholesterol (mg/dl)",
    min_value=100,
    max_value=600,
    value=200,
    help="Fat-like substances in blood; high levels increase risk.",
    key="chol"
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl?",
    options=[0,1],
    format_func=lambda x:"No" if x==0 else "Yes",
    help="Blood sugar over 120 after fasting may indicate diabetes.",
    key="fbs"
)

restecg = st.selectbox(
    "Resting ECG result",
    options=[0,1,2],
    format_func=lambda x:{
        0:"Normal",
        1:"ST-T abnormality",
        2:"Left ventricular hypertrophy"
    }[x],
    help="ECG test results indicating heart electrical activity.",
    key="restecg"
)

thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=60,
    max_value=220,
    value=140,
    help="Highest pulse during exercise or physical activity.",
    key="thalach"
)

exang = st.selectbox(
    "Exercise Induced Angina",
    options=[0,1],
    format_func=lambda x:"No" if x==0 else "Yes",
    help="Chest pain during physical activity.",
    key="exang"
)

oldpeak = st.number_input(
    "ST depression induced by exercise (oldpeak)",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    format="%.1f",
    help="Changes in heart's electrical pattern during exercise; higher may indicate stress.",
    key="oldpeak"
)

slope = st.selectbox(
    "Slope of the peak exercise ST segment",
    options=[0,1,2],
    format_func=lambda x:{
        0:"Upsloping",
        1:"Flat",
        2:"Downsloping"
    }[x],
    help=(
        "ECG signal pattern during peak exercise:\n"
        "- Upsloping (good)\n"
        "- Flat\n"
        "- Downsloping (may indicate risk)"
    ),
    key="slope"
)

ca = st.selectbox(
    "Number of major vessels colored by fluoroscopy (0-3)",
    options=[0,1,2,3],
    help="Number of heart vessels blocked (0 = none).",
    key="ca"
)

thal = st.selectbox(
    "Thalassemia status",
    options=[1,2,3],
    format_func=lambda x:{
        1:"Normal",
        2:"Fixed defect",
        3:"Reversible defect"
    }[x],
    help=(
        "Thalassemia status:\n"
        "- Normal\n"
        "- Fixed defect (permanent blockage/scar)\n"
        "- Reversible defect (blockage may improve)"
    ),
    key="thal"
)

# Collect user input vector in correct order for prediction
user_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                       exang, oldpeak, slope, ca, thal]])


# --- Prediction & Explainability ---
predict_clicked = st.button("Predict My Heart Disease Risk")

if predict_clicked:
    user_data_scaled = scaler.transform(user_data)
    pred_prob = model.predict_proba(user_data_scaled)[0][1]

    # Determine risk category
    if pred_prob >= 0.7:
        risk_category = "High Risk"
    elif pred_prob >= 0.4:
        risk_category = "Moderate Risk"
    else:
        risk_category = "Low Risk"

    # Store prediction and user data in session state (as 1D arrays)
    st.session_state['risk_category'] = risk_category
    st.session_state['pred_prob'] = pred_prob
    st.session_state['user_data_scaled'] = user_data_scaled.flatten()  # shape (13,)
    st.session_state['user_data_raw'] = user_data.flatten()            # shape (13,)

# Retrieve stored values outside of if-block so they persist between reruns
risk_category    = st.session_state.get('risk_category')
pred_prob        = st.session_state.get('pred_prob')
user_data_scaled = st.session_state.get('user_data_scaled')
user_data_raw    = st.session_state.get('user_data_raw')

# Only proceed if all necessary data exists
contributions = None
if (risk_category is not None) and (user_data_scaled is not None):
    coefs = model.coef_[0]
    contributions = coefs * np.array(user_data_scaled)

if (risk_category is not None) and (pred_prob is not None) \
   and (user_data_scaled is not None) and (user_data_raw is not None):

    st.markdown(f"### Your predicted heart disease risk is: **{risk_category}**")
    st.markdown(f"Probability score: **{pred_prob:.2f}**")

    # # Calculate top 3 contributing features
    # coefs = model.coef_[0]  # shape (13,)
    # contributions = coefs * user_data_scaled
    # abs_contrib = np.abs(contributions)
    # top_indices = abs_contrib.argsort()[-3:][::-1]

    # st.write("#### Top factors affecting your heart disease risk:")
    # for idx in top_indices:
    #     direction = "increase" if contributions[idx] > 0 else "decrease"
    #     st.write(f"- **{feature_names[idx]}** (value: {user_data_raw[idx]}): contributes to {direction} risk")

    # # Personalized static health recommendations
    # if risk_category == "High Risk":
    #     st.warning("You are at high risk for heart disease. Please consult a healthcare professional promptly.")
    #     st.markdown("""
    #     **Recommendations:**  
    #     - Follow your doctor's advice closely.  
    #     - Eat a heart-healthy diet: reduce salt and saturated fats; increase fruits and vegetables.  
    #     - Monitor blood pressure and cholesterol regularly.  
    #     - Avoid tobacco and limit alcohol intake.  
    #     - Engage in appropriate physical activity after consulting your doctor.
    #     """)
    # elif risk_category == "Moderate Risk":
    #     st.info("You have a moderate risk of heart disease. Consider these steps to lower your risk:")
    #     st.markdown("""
    #     **Recommendations:**  
    #     - Maintain a balanced diet rich in whole grains and vegetables.  
    #     - Aim for 150 minutes of moderate exercise weekly.  
    #     - Avoid smoking and limit alcohol.  
    #     - Monitor your blood pressure and cholesterol.
    #     """)
    # else:
    #     st.success("You are at low risk for heart disease. Keep up the good work!")
    #     st.markdown("""
    #     **Recommendations:**  
    #     - Continue healthy eating and regular exercise.  
    #     - Avoid smoking.  
    #     - Regular health screening is encouraged.
    #     """)
else:
    st.info("Please enter your details and click 'Predict My Heart Disease Risk'.")


# --- LLM Integration Section ---

st.markdown("---")

with st.expander("Open Chat with AI Health Assistant"):

    st.header("Personalized AI Health Assistant")

    # Prepare user_features_for_prompt dictionary
    user_features_for_prompt = {
        "Age": age,
        "Sex": "Male" if sex == 1 else "Female",
        "Chest Pain Type": {
            0: "Typical angina",
            1: "Atypical angina",
            2: "Non-anginal pain",
            3: "No chest pain"
        }[cp],
        "Resting Blood Pressure": trestbps,
        "Cholesterol": chol,
        "Fasting Blood Sugar > 120 mg/dl": "Yes" if fbs == 1 else "No",
        "Resting ECG": {
            0: "Normal",
            1: "ST-T abnormality",
            2: "Left ventricular hypertrophy"
        }[restecg],
        "Max Heart Rate": thalach,
        "Exercise Induced Angina": "Yes" if exang == 1 else "No",
        "ST depression": oldpeak,
        "Slope": {
            0: "Upsloping",
            1: "Flat",
            2: "Downsloping"
        }[slope],
        "Major Vessels": ca,
        "Thalassemia": {
            1: "Normal",
            2: "Fixed defect",
            3: "Reversible defect"
        }[thal]
    }

    # Retrieve stored prediction-related variables
    risk_category = st.session_state.get('risk_category')
    pred_prob = st.session_state.get('pred_prob')
    user_data_scaled = st.session_state.get('user_data_scaled')

    # Compute contributions for abnormal features detection
    contributions = None
    if (risk_category is not None) and (user_data_scaled is not None):
        coefs = model.coef_[0]
        contributions = coefs * np.array(user_data_scaled)

    def find_abnormal_features():
        abnormal_feats = []
        threshold = 0.1  # Adjust threshold as needed
        if contributions is not None:
            for i, val in enumerate(contributions):
                if abs(val) >= threshold:
                    abnormal_feats.append(feature_names[i])
        return abnormal_feats

    # Initialize user_answers dict for storing user's lifestyle answers
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}

    # Helper to summarize user answers for prompt
    def summarize_user_answers():
        if not st.session_state.user_answers:
            return "No lifestyle information provided yet."
        lines = []
        for topic, answer in st.session_state.user_answers.items():
            if isinstance(answer, list):
                # Join multiple answers if stored as list
                lines.append(f"- {topic.capitalize()}: {', '.join(answer)}")
            else:
                lines.append(f"- {topic.capitalize()}: {answer}")
        return "\n".join(lines)

    # Determine AI tone instructions based on risk level
    if risk_category == "Low Risk":
        tone_instruction = (
            "Your tone should be warm, very encouraging, and positive. "
            "Congratulate the user for having a low heart risk, and motivate them to maintain their healthy lifestyle."
        )
    elif risk_category == "Moderate Risk":
        tone_instruction = (
            "Your tone should be calm, informative, and slightly cautionary. "
            "Speak in a balanced and supportive way, suggesting lifestyle improvements to lower risk."
        )
    elif risk_category == "High Risk":
        tone_instruction = (
            "Your tone should be serious, strict, and urgent, but empathetic. "
            "Make it clear the user is at high heart risk and must take immediate action, offering clear, actionable steps."
        )
    else:
        tone_instruction = "Use a neutral and empathetic tone."

    if risk_category is None:
        st.info("Please perform a risk prediction first to get personalized AI advice.")
    else:
        if 'llm_response' not in st.session_state:
            st.session_state.llm_response = ""
        if "has_greeted" not in st.session_state:
            st.session_state.has_greeted = False

        # Button to get initial personalized advice and start conversation
        if st.button("Get AI Personalised Advice"):
            abnormal_features = find_abnormal_features()

            prompt = f"""You are a compassionate heart health assistant.

The user has a heart disease risk level: {risk_category} (probability: {pred_prob:.2f}).

User's health features:
"""
            for k, v in user_features_for_prompt.items():
                prompt += f"- {k}: {v}\n"

            if risk_category in ["Moderate Risk", "High Risk"]:
                prompt += f"\nThe following health features are abnormal and need special attention: {', '.join(abnormal_features)}.\n"
            else:
                prompt += "\nThe user is at low risk; appreciate and encourage healthy habits.\n"

            prompt += "\nUser lifestyle information so far:\n"
            prompt += summarize_user_answers()
            prompt += "\n\nInstructions:\n"
            prompt += (
                f"{tone_instruction}\n"
                "Start the conversation with a warm, compassionate greeting only if this is the first message.\n"
                "Then, ask open-ended questions to understand the user's lifestyle, for example: "
                "‘Could you tell me about your smoking habits?’, ‘What kind of diet do you usually follow?’, ‘How do you manage stress?’ "
                "Cover topics like smoking, diet, stress, family history of heart disease, exercise, alcohol consumption, and sleep.\n"
                "Use the user's previous answers to personalize your advice and avoid repeating questions or greetings.\n"
                "If all relevant topics are covered, summarize your advice and suggest actionable next steps.\n"
                "Answer conversationally and naturally, avoiding repetitive introductions.\n"
            )

            # Only greet in first AI message
            if not st.session_state.has_greeted:
                prompt += "\nGreeting: Hi! I'm here to help you understand and manage your heart disease risk.\n"
                st.session_state.has_greeted = True

            try:
                response = gemini_model.generate_content(prompt)
                st.session_state.llm_response = response.text

                # Initialize chat history only once with this AI initial response
                if "chat_history" not in st.session_state or len(st.session_state.chat_history) == 0:
                    st.session_state.chat_history = [{"user": "", "ai": response.text}]
            except Exception as e:
                st.error(f"Error communicating with Gemini API: {e}")

        # Display initial LLM response and chat only if no chat yet
        if st.session_state.llm_response and ("chat_history" not in st.session_state or len(st.session_state.chat_history) == 0):
            st.markdown("#### AI Health Assistant Response:")
            st.write(st.session_state.llm_response)

    # Initialize chat history if absent
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages with visual distinction
    for entry in st.session_state.chat_history:
        if entry.get("user"):
            st.chat_message("user").markdown(entry["user"])
        if entry.get("ai"):
            st.chat_message("assistant").markdown(entry["ai"])

    # User input box to continue conversation
    user_message = st.chat_input("Your reply...")

    if user_message:
        # Append user's message to chat history
        st.session_state.chat_history.append({"user": user_message, "ai": ""})

        # Try to infer lifestyle topic from last AI question, saved in user_answers
        last_ai_msg = ""
        for turn in reversed(st.session_state.chat_history):
            if turn.get("ai"):
                last_ai_msg = turn["ai"]
                break

        # Keywords for heuristic topic matching
        lifestyle_topics = {
            "smoking": ["smok", "cigarette", "tobacco"],
            "diet": ["diet", "food", "eat", "nutrition"],
            "stress": ["stress", "anxiety", "pressure"],
            "family history": ["family", "hereditary", "relative"],
            "exercise": ["exercise", "physical activity", "workout", "fitness"],
            "alcohol": ["alcohol", "drink", "wine", "beer", "liquor"],
            "sleep": ["sleep", "rest", "insomnia"]
        }

        last_ai_msg_lc = last_ai_msg.lower()
        user_msg_lc = user_message.lower()

        answered_topic = None
        for topic, keywords in lifestyle_topics.items():
            if any(kw in last_ai_msg_lc for kw in keywords):
                answered_topic = topic
                break

        if answered_topic:
            st.session_state.user_answers[answered_topic] = user_message
        else:
            # Store under 'general' if no topic matched
            if "general" not in st.session_state.user_answers:
                st.session_state.user_answers["general"] = []
            st.session_state.user_answers["general"].append(user_message)

        # Build prompt including full context for next LLM call
        def build_chat_prompt():
            prompt = "You are a compassionate heart health assistant.\n"
            prompt += f"Risk level: {risk_category} (probability: {pred_prob:.2f})\n"
            prompt += "User's health features:\n"
            for k, v in user_features_for_prompt.items():
                prompt += f"- {k}: {v}\n"
            prompt += "\nUser lifestyle information collected so far:\n"
            prompt += summarize_user_answers() + "\n\n"
            prompt += "Conversation history:\n"
            for turn in st.session_state.chat_history:
                if turn.get("user"):
                    prompt += f"User: {turn['user']}\n"
                if turn.get("ai"):
                    prompt += f"AI: {turn['ai']}\n"
            prompt += f"\nTone: {tone_instruction}\n"
            prompt += (
                "Provide empathetic advice tailored to the user's heart risk and previous answers. "
                "Ask the next open-ended lifestyle question without repeating greetings. "
                "Keep the language simple, clear, and friendly.\n"
            )
            return prompt

        prompt = build_chat_prompt()

        try:
            response = gemini_model.generate_content(prompt)
            # Update last AI message in chat history
            st.session_state.chat_history[-1]["ai"] = response.text
        except Exception as e:
            st.error(f"Error communicating with Gemini API: {e}")

        # Rerun app to update UI and show new messages
        st.rerun()


