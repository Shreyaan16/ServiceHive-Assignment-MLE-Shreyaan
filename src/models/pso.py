from utils.pso_req import load_data , prepare_data , PSO , create_model , train_model , evaluate_model
from sklearn.utils import shuffle 

(train_images, train_labels), (test_images, test_labels) = load_data()
train_images, train_labels = shuffle(train_images, train_labels, random_state=25)

train_images = train_images / 255.0 
test_images = test_images / 255.0

n_train = train_labels.shape[0]
n_test = test_labels.shape[0]
train_dataloader, test_dataloader = prepare_data(train_images, train_labels, 
                                                   test_images, test_labels, batch_size=64)

bounds = [(0.0001, 0.01), (16, 64)]

best_params, best_score = PSO(bounds=bounds, train_dataloader=train_dataloader, 
                                 n_particles=10, max_iter=2)

final_model, final_optimizer, final_criterion = create_model(
        learning_rate=best_params[0], 
        num_filters=int(best_params[1])
    )

final_accuracies = train_model(final_model, final_optimizer, final_criterion, 
                                 train_dataloader, epochs=20, verbose=True)

test_accuracy = evaluate_model(final_model, test_dataloader)