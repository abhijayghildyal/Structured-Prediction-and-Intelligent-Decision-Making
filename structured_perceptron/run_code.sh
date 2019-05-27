python3 -u structured_perceptron_train.py --dataset nettalk_stress >> nettalk_train_output.out
python3 -u structured_perceptron_test.py --dataset nettalk_stress >> nettalk_test_output.out
python3 -u structured_perceptron_train.py --dataset ocr_fold0_sm >> ocr_train_output.out
python3 -u structured_perceptron_test.py --dataset ocr_fold0_sm >> ocr_test_output.out