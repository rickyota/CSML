# bash file to execute CSML

# for training classifier and inferring images.
python -m csml.src \
    --train "./data/train/" \
    --label "./data/label/" \
    --output "./result/example/" \
    --model "./data/model.pkl" \
    --infer "./data/infer/" \

