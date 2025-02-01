# 🎓 Student Recommendations System

## 📌 Overview
The **Student Recommendations System** is an AI-powered model that analyzes student performance data to provide personalized insights. It helps students identify weak areas, track improvements, detect performance gaps, and receive tailored recommendations to enhance learning outcomes.

## 🚀 Features
- **Weak Areas Analysis** – Identifies subjects requiring more focus using `correct_answers`, `incorrect_answers`, `mistakes_corrected` with KMeans clustering.
- **Improvement Tracking** – Monitors student progress over time using `rank` and `accuracy` with a GaussianMixture model.
- **Performance Gaps Detection** – Highlights areas that need more attention using `negative_score`, `initial_mistake_count`, `total_questions`, and `rank` with KMeans clustering.
- **Personalized Recommendations** – Suggests strategies for better learning using `better_than` and `trophy_level`.
- **Student Persona Classification** – Categorizes students based on `speed`, `rank`, and `source` using KMeans clustering.

## 🛠️ Technologies Used
- **Python** – Core programming language
- **Scikit-Learn** – Machine learning models
- **Pandas & NumPy** – Data processing and manipulation
- **Pickle** – Model storage and loading

## 🔧 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/student-recommendation-system.git
   cd student-recommendation-system
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have trained models stored as:
   - `weak_areas_model.pt`
   - `improvements_model.pt`
   - `performance_gaps_model.pt`
   - `recommendations_model.pt`
   - `student_persona_model.pt`

## 📊 Data Processing
- **Dataset Preparation**:
  - Load the dataset and remove irrelevant features.
  - Convert start and end times into a `time_consumed_percentage` feature.
  - Update accuracy using the final score and provided score details.
  - Convert rank into numerical format.

## 📊 Usage
1. Load the models and run predictions:
   ```python
   from studentrecommendationsystem import predict_weak_areas, predict_improvements, predict_performance_gaps, predict_recommendations, predict_student_persona

   real_world_data = {
       "weak_areas": [[15, 10, 5]],
       "improvements": [[85, 3]],
       "performance_gaps": [[3, 20, 100, 5]],
       "recommendations": [[75, 2]],
       "student_persona": [[120, 5, 0.3333]]
   }

   print(predict_weak_areas(real_world_data["weak_areas"]))
   print(predict_improvements(real_world_data["improvements"]))
   print(predict_performance_gaps(real_world_data["performance_gaps"]))
   print(predict_recommendations(real_world_data["recommendations"]))
   print(predict_student_persona(real_world_data["student_persona"]))
   ```

2. The system will return personalized recommendations based on the input data.

## 📌 Contribution
Feel free to fork the repository, create a new branch, and submit a pull request for any improvements!

## 📄 License
This project is open-source under the **MIT License**.

---

🔗 **[Demo or Deployment Link (If Available)]**
