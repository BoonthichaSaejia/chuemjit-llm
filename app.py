from flask import Flask, request, jsonify
from chuemjit import get_ans, gen_text, replace_text

app = Flask(__name__)

@app.route('/api/query', methods=['GET'])
def handle_query():
    query = request.args.get('query')
    threshold = float(request.args.get('threshold', 0.8))  # Default threshold 0.8 if not provided

    # Process the query and generate a response
    list_text = get_ans(query, threshold)
    prompt = f"จงแปลเป็นภาษาอีสาน {list_text}\n11.) ไทยกลาง: {query} ,อีสาน:"
    new_isan = gen_text(prompt)
    translated_text = replace_text(new_isan)

    # Prepare response as JSON
    response = {
        'query': query,
        'translated_text': translated_text
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)