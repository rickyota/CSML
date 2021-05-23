# bash file to execute CSML 

# only infer using model trained before
python -m projects.csml.src \
    --finfer
    --output "./result/example/" \
    --model "./data/model.pkl" \
    --infer "./data/infer/" 


