import streamlit as st
from streamlit_option_menu import option_menu
from pandas import DataFrame,read_csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense


salary_data = read_csv('content/salary.csv')

null_indices = salary_data[salary_data.isnull().any(axis=1)].index.tolist()

salary_data = salary_data.drop(index=null_indices,axis=1)

target_salary_data = salary_data['Salary']
salary_data = salary_data.drop(columns=['Salary'],axis=1)

def build_model(input_dim):
    model = Sequential()
    model.add(Dense(147, activation='relu', input_dim=input_dim))
    model.add(Dense(45, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer='adam', loss='huber', metrics=['mae'])
    return model

class TrainAndEvaluate(BaseEstimator, TransformerMixin):
    def __init__(self, model_builder, test_size=0.2, random_state=42):
        self.model_builder = model_builder
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, salary_data, target_salary_data):
        self.train_salary_data, self.test_salary_data, self.train_target_salary_data, self.test_target_salary_data = train_test_split(
            salary_data, target_salary_data, test_size=self.test_size, random_state=self.random_state
        )
        self.model = self.model_builder(salary_data.shape[1])
        self.model.fit(self.train_salary_data, self.train_target_salary_data, epochs=100, batch_size=32, verbose=0)
        return self

    def predict(self, X=None):
        if X is None:
            return self.model.predict(self.test_salary_data)
        return self.model.predict(X)

    def score(self, X=None, y=None):
        if X is None or y is None:
            X, y = self.test_salary_data, self.test_target_salary_data
        return self.model.evaluate(X, y, verbose=0)[1]

categorical_cols = ["Education Level", "Job Title", "Gender"]
numerical_cols = ["Age", "Years of Experience"]

preprocessor = ColumnTransformer([
    ("num", MinMaxScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("pca", PCA(n_components=0.95)),
    ("model_trainer", TrainAndEvaluate(build_model))
])


pipeline.fit(salary_data, target_salary_data)

with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Model Description", "Model Execution"],
        icons=["info-circle", "gear"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical"
    )


if selected == "Model Description":
    st.title("🧠 Model Description")
    st.markdown("""
    This app demonstrates a simple ML model.

    **Details:**
    - Model Type: ANN or Regression
    - Dataset: Salary, MNIST, etc.
    - Purpose: Predict salary / classify digit / other logic
    """)

elif selected == "Model Execution":
    pass
gender_options = ["Male", "Female"]

age_options = [str(i) for i in range(23, 54)]

education_options = ["Bachelor's", "Master's", "PhD"]


job_titles = [
    'Account Manager', 'Accountant', 'Administrative Assistant', 'Business Analyst',
    'Business Development Manager', 'Business Intelligence Analyst', 'CEO',
    'Chief Data Officer', 'Chief Technology Officer', 'Content Marketing Manager',
    'Copywriter', 'Creative Director', 'Customer Service Manager', 'Customer Service Rep',
    'Customer Service Representative', 'Customer Success Manager', 'Customer Success Rep',
    'Data Analyst', 'Data Entry Clerk', 'Data Scientist', 'Digital Content Producer',
    'Digital Marketing Manager', 'Director', 'Director of Business Development',
    'Director of Engineering', 'Director of Finance', 'Director of HR',
    'Director of Human Capital', 'Director of Human Resources', 'Director of Marketing',
    'Director of Operations', 'Director of Product Management', 'Director of Sales',
    'Director of Sales and Marketing', 'Event Coordinator', 'Financial Advisor',
    'Financial Analyst', 'Financial Manager', 'Graphic Designer', 'HR Generalist',
    'HR Manager', 'Help Desk Analyst', 'Human Resources Director', 'IT Manager',
    'IT Support', 'IT Support Specialist', 'Junior Account Manager', 'Junior Accountant',
    'Junior Advertising Coordinator', 'Junior Business Analyst',
    'Junior Business Development Associate', 'Junior Business Operations Analyst',
    'Junior Copywriter', 'Junior Customer Support Specialist', 'Junior Data Analyst',
    'Junior Data Scientist', 'Junior Designer', 'Junior Developer',
    'Junior Financial Advisor', 'Junior Financial Analyst', 'Junior HR Coordinator',
    'Junior HR Generalist', 'Junior Marketing Analyst', 'Junior Marketing Coordinator',
    'Junior Marketing Manager', 'Junior Marketing Specialist', 'Junior Operations Analyst',
    'Junior Operations Coordinator', 'Junior Operations Manager', 'Junior Product Manager',
    'Junior Project Manager', 'Junior Recruiter', 'Junior Research Scientist',
    'Junior Sales Representative', 'Junior Social Media Manager',
    'Junior Social Media Specialist', 'Junior Software Developer',
    'Junior Software Engineer', 'Junior UX Designer', 'Junior Web Designer',
    'Junior Web Developer', 'Marketing Analyst', 'Marketing Coordinator',
    'Marketing Manager', 'Marketing Specialist', 'Network Engineer', 'Office Manager',
    'Operations Analyst', 'Operations Director', 'Operations Manager',
    'Principal Engineer', 'Principal Scientist', 'Product Designer', 'Product Manager',
    'Product Marketing Manager', 'Project Engineer', 'Project Manager',
    'Public Relations Manager', 'Recruiter', 'Research Director', 'Research Scientist',
    'Sales Associate', 'Sales Director', 'Sales Executive', 'Sales Manager',
    'Sales Operations Manager', 'Sales Representative', 'Senior Account Executive',
    'Senior Account Manager', 'Senior Accountant', 'Senior Business Analyst',
    'Senior Business Development Manager', 'Senior Consultant', 'Senior Data Analyst',
    'Senior Data Engineer', 'Senior Data Scientist', 'Senior Engineer',
    'Senior Financial Advisor', 'Senior Financial Analyst', 'Senior Financial Manager',
    'Senior Graphic Designer', 'Senior HR Generalist', 'Senior HR Manager',
    'Senior HR Specialist', 'Senior Human Resources Coordinator',
    'Senior Human Resources Manager', 'Senior Human Resources Specialist',
    'Senior IT Consultant', 'Senior IT Project Manager', 'Senior IT Support Specialist',
    'Senior Manager', 'Senior Marketing Analyst', 'Senior Marketing Coordinator',
    'Senior Marketing Director', 'Senior Marketing Manager',
    'Senior Marketing Specialist', 'Senior Operations Analyst',
    'Senior Operations Coordinator', 'Senior Operations Manager',
    'Senior Product Designer', 'Senior Product Development Manager',
    'Senior Product Manager', 'Senior Product Marketing Manager',
    'Senior Project Coordinator', 'Senior Project Manager',
    'Senior Quality Assurance Analyst', 'Senior Research Scientist', 'Senior Researcher',
    'Senior Sales Manager', 'Senior Sales Representative', 'Senior Scientist',
    'Senior Software Architect', 'Senior Software Developer', 'Senior Software Engineer',
    'Senior Training Specialist', 'Senior UX Designer', 'Social Media Manager',
    'Social Media Specialist', 'Software Developer', 'Software Engineer',
    'Software Manager', 'Software Project Manager', 'Strategy Consultant',
    'Supply Chain Analyst', 'Supply Chain Manager', 'Technical Recruiter',
    'Technical Support Specialist', 'Technical Writer', 'Training Specialist',
    'UX Designer', 'UX Researcher', 'VP of Finance', 'VP of Operations',
    'Web Developer'
]

with st.form("input_form"):
    st.title("📝 Candidate Info Form")

    gender = st.selectbox("Gender", gender_options)
    age = st.selectbox("Age", age_options)
    education = st.selectbox("Education Level", education_options)
    job_title = st.selectbox("Job Title", job_titles)
    experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.5)

    submit_btn = st.form_submit_button("Submit")

if submit_btn:
    st.success("✅ Form submitted successfully!")


    data = prepare_input_for_prediction(experience,age,education,job_title,gender)

    print(data)
    prediction = model.predict(data)
    prediction = target_scaler.inverse_transform(prediction.reshape(-1, 1))

    st.markdown(f"""
    **Your Input Summary:**
    - 👤 Gender: `{gender}`
    - 🎂 Age: `{age}`
    - 🎓 Education: `{education}`
    - 💼 Job Title: `{job_title}`
    - 🧮 Experience: `{experience} years`
    - 🔍 Model Prediction: `{prediction}`
    """)


