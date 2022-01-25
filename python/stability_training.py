from curses import nocbreak
from function_utils import initialize_model, accuracies_arrays
from class_utils import Model_mode
import torch
import torch.nn as nn
import transforms as T
from torchvision import transforms
import time
import datetime
from datasets import AWGNImageNetDataset
from terminaltables import AsciiTable
from PIL import Image
from decimal import Decimal
import os

#This script is an implementation of the paper called "Improving the robustness of deep neural networks via stability training".

################################################################################################################
##########          HYPERPARAMETERS          ###################################################################
################################################################################################################

epsilon = 1e-8
print_density = 100

hardware = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(hardware)

img_base_width_height = 224

num_workers = 16

num_epochs = 45
conv_layers_fixed = True
learning_rate_decay_epoch_not_beaten = 3
learning_rate_decay_factor = 5.0
stopping_learning_rate_base = 1/learning_rate_decay_factor**1 #If the learning rate got divided too much times (here, 1 time), the training is stopped

################################################################################################################
##########          MAIN          ##############################################################################
################################################################################################################

def main(model_mode: int, learning_rate, lambda_loss, uniform_gaussian_noise_std):
    stopping_learning_rate = stopping_learning_rate_base*learning_rate
    
    gpu_batch_size = 384 if model_mode == Model_mode.resnet50 else 320 if model_mode == Model_mode.mnasnet else -1
    batch_size = 1 if hardware == "cpu" else gpu_batch_size

    folder_string = "lr_" + '%.2E' % Decimal(learning_rate) + "_lambda_loss_" + '%.2E' % Decimal(lambda_loss) + "_AWGN_std_" + '%.2E' % Decimal(uniform_gaussian_noise_std) + "/"
    os.makedirs("/weights/" + folder_string, exist_ok=True)
    os.makedirs("/csv/" + folder_string, exist_ok=True)

    #Initialize model
    model, input_size = initialize_model(Model_mode(model_mode).name, True, requires_grad=True, conv_layers_fixed=conv_layers_fixed)

    #Run on multiple GPUs
    model = nn.DataParallel(model).to(device)

    params_to_update = model.parameters()
    if conv_layers_fixed:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
    
    optimizer = torch.optim.Adam(params_to_update, betas=(0.5, 0.99), lr=learning_rate)
    
    #Initialize loss
    classification_loss_fn = nn.CrossEntropyLoss()
    stability_loss_fn = nn.KLDivLoss(reduction='sum')

    if input_size < img_base_width_height:
        #This is done because we do not want the square image to correspond to a small fraction of the original image
        preprocess_transform = transforms.Compose([transforms.Resize(256, interpolation=Image.BICUBIC),
                                                   transforms.CenterCrop(img_base_width_height),
                                                   transforms.Resize(input_size, interpolation=Image.BICUBIC)])
    elif input_size == img_base_width_height:
        preprocess_transform = transforms.Compose([transforms.Resize(256, interpolation=Image.BICUBIC),
                                                   transforms.CenterCrop(input_size)])
    else: #input_size >= img_base_width_height
        preprocess_transform = transforms.Compose([transforms.Resize(int(256*(input_size/img_base_width_height)), interpolation=Image.BICUBIC),
                                                   transforms.CenterCrop(int(input_size))])
                
    mean_norm, std_norm = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32), torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    postprocess_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean_norm.tolist(), std=std_norm.tolist())])
    
    datasets = {'train' : AWGNImageNetDataset("/data/imagenet/", 'train', uniform_gaussian_noise_std, preprocess_transform, postprocess_transform),
                'val'   : AWGNImageNetDataset("/data/imagenet/", 'val'  , uniform_gaussian_noise_std, preprocess_transform, postprocess_transform)}

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True) for x in ['train', 'val']}    
    
    lowest_mean_val_loss_obtained, lowest_mean_val_loss_not_beaten_counter = 727, 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        for phase in ['train', 'val']:
            if epoch == 0 and phase == 'train':
                continue

            start_time = time.time()
            if phase == 'train':
                print("\n---- TRAINING ----")
                model.train()
            else:
                print("\n---- VALIDATION ----")
                model.eval()
            
            #Variables for logs
            batch_i = 0
            accumulated_classification_loss, accumulated_stability_loss, accumulated_final_loss = 0.0, 0.0, 0.0
            imagenet_correctly_classified_samples_top_1, imagenet_correctly_classified_samples_top_5 = 0, 0
            
            for _, undistorted_batch, distorted_batch, labels_batch in dataloaders[phase]:
                if batch_i % (5*print_density) == 0:
                    print("Epoch " + str(epoch), " / ", str(num_epochs), "    |   ", "Batch " + str(batch_i), " / ", str(len(dataloaders[phase])))
                batch_i += 1
                
                undistorted_batch, distorted_batch, labels_batch = undistorted_batch.to(device), distorted_batch.to(device), labels_batch.to(device)

                ###############################################################################################################
                ##########          Compute classification and stability loss          ########################################
                ###############################################################################################################
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(False):
                    distorted_output = model(distorted_batch)
                
                with torch.set_grad_enabled(phase == 'train'):
                    undistorted_output = model(undistorted_batch)
                
                    classification_loss = classification_loss_fn(undistorted_output, labels_batch)
                    stability_loss = lambda_loss*stability_loss_fn(nn.functional.log_softmax(distorted_output, dim=1), nn.functional.softmax(undistorted_output, dim=1))
                    final_loss = classification_loss + stability_loss

                ###############################################################################################################
                ##########          Backpropagation          ##################################################################
                ###############################################################################################################
                
                if phase == 'train':
                    final_loss.backward()
                    optimizer.step()   

                ###############################################################################################################
                
                top_1_accuracies, top_5_accuracies = accuracies_arrays(distorted_output, labels_batch.data, topk=(1,5))
                
                with torch.set_grad_enabled(False):
                    accumulated_classification_loss += classification_loss.item() * len(labels_batch)
                    accumulated_stability_loss += stability_loss.item() * len(labels_batch)
                    accumulated_final_loss += final_loss.item() * len(labels_batch)
                    
                    imagenet_correctly_classified_samples_top_1  += top_1_accuracies.sum().item()
                    imagenet_correctly_classified_samples_top_5  += top_5_accuracies.sum().item()

                    if batch_i % print_density == 0:
                        epoch_batches_left = len(dataloaders[phase]) - (batch_i + 1)
                        time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                        print(f"    ---- ETA {time_left}")

            ###############################################################################################################
            ##########          END OF EPOCH          #####################################################################
            ###############################################################################################################
            
            accumulated_classification_loss = accumulated_classification_loss / (epsilon + len(datasets[phase]))
            accumulated_stability_loss = accumulated_stability_loss / (epsilon + len(datasets[phase]))
            accumulated_final_loss = accumulated_final_loss / (epsilon + len(datasets[phase]))

            top_1_compressed_accuracy =  imagenet_correctly_classified_samples_top_1 / (epsilon + float(len(datasets[phase])))
            top_5_compressed_accuracy =  imagenet_correctly_classified_samples_top_5 / (epsilon + float(len(datasets[phase])))
            
            table =  [["Metrics (" + ("TRAINING" if phase == 'train' else "VALIDATION") + " set)", "values"]]
            table += [["learning rate", "%.8f" % learning_rate]]
            table += [["classification loss", "%.8f" % accumulated_classification_loss]]
            table += [["stability loss", "%.8f" % accumulated_stability_loss]]
            table += [["final loss", "%.8f" % accumulated_final_loss]]
            table += [["AWGN std", uniform_gaussian_noise_std]]
            table += [["top-1 compressed accuracy",  "%.8f" % top_1_compressed_accuracy]]
            table += [["top-5 compressed accuracy",  "%.8f" % top_5_compressed_accuracy]]
            print(AsciiTable(table).table)

            #Saving csv files containing validation and training loss at every epoch, s.t. we can plot it afterwards
            csv_file = open("/csv/" + folder_string + "classification_loss__" + phase + "_loss__epoch_" + str(epoch) + ".csv", "w")
            csv_file.write(str(epoch) + "\n" + str(accumulated_classification_loss))
            csv_file.close()
            csv_file = open("/csv/" + folder_string + "stability_loss__" + phase + "_loss__epoch_" + str(epoch) + ".csv", "w")
            csv_file.write(str(epoch) + "\n" + str(accumulated_stability_loss))
            csv_file.close()
            csv_file = open("/csv/" + folder_string + "final_loss__" + phase + "_loss__epoch_" + str(epoch) + ".csv", "w")
            csv_file.write(str(epoch) + "\n" + str(accumulated_final_loss))
            csv_file.close()
            csv_file = open("/csv/" + folder_string + "Accuracy_top_1__" + phase + "_accuracy__epoch_" + str(epoch) + ".csv", "w")
            csv_file.write(str(epoch) + "\n" + str(top_1_compressed_accuracy))
            csv_file.close()
            csv_file = open("/csv/" + folder_string + "Accuracy_top_5__" + phase + "_accuracy__epoch_" + str(epoch) + ".csv", "w")
            csv_file.write(str(epoch) + "\n" + str(top_5_compressed_accuracy))
            csv_file.close()

            #Decreasing learning rate if validation loss was not beaten in the last few epochs
            if not phase == 'train' and epoch != 0:
                if lowest_mean_val_loss_obtained > accumulated_final_loss:
                    lowest_mean_val_loss_obtained = accumulated_final_loss
                    lowest_mean_val_loss_not_beaten_counter = 0
                else:
                    if lowest_mean_val_loss_not_beaten_counter == learning_rate_decay_epoch_not_beaten:
                        lowest_mean_val_loss_not_beaten_counter = 0
                        learning_rate = learning_rate / learning_rate_decay_factor

                        if learning_rate <= stopping_learning_rate:
                            print("Current learning rate is : ", learning_rate)
                            print("Stopping...")
                            return 0 #Early stopping
                        
                        params_to_update = model.parameters()
                        if conv_layers_fixed:
                            params_to_update = []
                            for name,param in model.named_parameters():
                                if param.requires_grad:
                                    params_to_update.append(param)
                        optimizer = torch.optim.Adam(params_to_update, betas=(0.5, 0.99), lr=learning_rate)
                    else:
                        lowest_mean_val_loss_not_beaten_counter += 1

            if phase == 'train':
                torch.save(model.state_dict(), "/weights/" + folder_string + "model_wts_epoch_" + str(epoch) + ".pth")
                
################################################################################################################

if __name__ == "__main__":
    #main(Model_mode.mnasnet , learning_rate = 1e-6, lambda_loss = 5e-3, uniform_gaussian_noise_std = 4e-2)
    main(Model_mode.resnet50, learning_rate = 1e-7, lambda_loss = 5e-3, uniform_gaussian_noise_std = 5e-4)