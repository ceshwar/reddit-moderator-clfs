import pandas as pd
import subprocess

print("Get predictions from pre-trained exports on test comments!")

test_comments = pd.read_csv('test_comments.csv', names = ['comment'])
count = 1

for study_sub in pd.read_csv('study-subreddits.csv', names= ['subreddit'])['subreddit']:
    print(count, ") Expert: " , study_sub)
    count+=1
    command = ["./fasttext", "predict", "pretrained_models/model_" + study_sub + ".bin", "test_comments.csv", "1"]
    result = subprocess.run(command, stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    expert_decision = output.split('\n')[:-1]
    ###store each experts prediction as a new column, containing the subreddit's name
    test_comments['prediction_' + study_sub] = expert_decision
    test_comments['prediction_' + study_sub] = (test_comments['prediction_' + study_sub] == '__label__removed').astype(int)

    if count % 50 == 0:
        print("Writing to file...")
        test_comments.to_csv(str(count) + '-experts-predictions-all-removed.csv', index = None)

test_comments.to_csv('all-experts-predictions-all-removed.csv', index = None)

print("Predictions from pre-trained exports on test comments = COMPLETE!")


