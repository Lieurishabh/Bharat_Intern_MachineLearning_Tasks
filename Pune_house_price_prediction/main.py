from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('final_pune_house_data.csv')
pipe = pickle.load(open("RFmodel.pkl", 'rb'))

@app.route('/')
def index():
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    balconies = sorted(data['balcony'].unique())
    area_list = sorted(data['area_type'].unique())
    size_list = sorted(data['size'].unique())
    

    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, balconies=balconies, area_list=area_list, size_list=size_list)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the form data and convert to appropriate types
        bedrooms = request.form.get('beds')
        bathrooms = request.form.get('baths')
        balconies = request.form.get('balcony')
        area_type = request.form.get('area_type')  # Keep as string for encoding
        size = request.form.get('size')

        # Create a DataFrame with the input data
        input_data = pd.DataFrame([[bedrooms, bathrooms, balconies, area_type, size]],
                                   columns=['beds', 'baths', 'balcony', 'area_type', 'size'])

        print("Input Data:")
        print(input_data)
        
            # Convert 'baths' column to numeric with errors='coerce'
        input_data['baths'] = pd.to_numeric(input_data['baths'], errors='coerce')

        # Convert input data to numeric types
        input_data = input_data.astype({'beds': float, 'baths': float, 'balcony': float, 'area_type': int, 'size': float})

        # Handle unknown categories in the input data
        for column in input_data.columns:
            unknown_categories = set(input_data[column]) - set(data[column].unique())
            if unknown_categories:
                print(f"Unknown categories in {column}: {unknown_categories}")
                # Handle unknown categories (e.g., replace with a default value)
                input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])

        print("Processed Input Data:")
        print(input_data)

        # Predict the price
        prediction = pipe.predict(input_data)[0]
        print(prediction)

        return str(prediction)
    
    except Exception as e:
        return f"An error occurred: {str(e)}"



if __name__ == "__main__":
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    app.run(debug=True, port=5000)