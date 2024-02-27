from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        text = request.form['text']
        # Assuming you have a TF-IDF vectorizer fit during training
        new_text_vectorized = vectorizer.transform([text])
        prediction = model.predict(new_text_vectorized)

        return render_template('result.html', text=text, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)