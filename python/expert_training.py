from function_utils import initialize_model, accuracies_arrays, get_distortion_mean_rate, initialize_datasets_dataloader
from class_utils import Model_mode, Distortion_mode, Distortion, Distortion_precompute
import torch
import torch.nn as nn
import os
import time
import datetime
from terminaltables import AsciiTable
import argparse

################################################################################################################
##########          HYPERPARAMETERS          ###################################################################
################################################################################################################

epsilon = 1e-8 #Used to avoid division by 0
print_density = 100

hardware = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(hardware)

img_base_width_height = 224

num_workers = 16

num_epochs = 45
start_epoch = -1 #Default value is -1 to start with an evaluation step using pretrained weights (on images without added coding artifacts)
conv_layers_fixed = False #As opposed to stability training paper, the whole model is trained (including convolutional layers)
learning_rate_decay_epoch_not_beaten = 2 #If the validation loss is not beaten within this number of epoch, then the learning rate is divided by learning_rate_decay_factor 
learning_rate_decay_factor = 5.0
stopping_learning_rate_base = 1/learning_rate_decay_factor**3 #If the learning rate got divided too much times (here, 3 times), the training is stopped

################################################################################################################
##########          MAIN          ##############################################################################
################################################################################################################

def main(model_mode: int, init_weights, distortion: Distortion, distortion_precompute_mode: int, is_fine_tuning: bool, learning_rate, alpha):
    stopping_learning_rate = stopping_learning_rate_base*learning_rate

    gpu_batch_size = 384 if model_mode == Model_mode.resnet50 else 320 if model_mode == Model_mode.mnasnet else -1
    batch_size = 1 if hardware == "cpu" else gpu_batch_size

    undistorted_output_path = "/data/_preprocess_undistorted_output/" + Model_mode(model_mode).name + "/" #Used for precomputing of model on undistorted samples
    is_undistorted_output_precomputed = os.path.exists(undistorted_output_path + 'val' + "/") and os.path.exists(undistorted_output_path + 'train' + "/")

    #Initialize models
    model, input_size = initialize_model(Model_mode(model_mode).name, True, requires_grad=True, conv_layers_fixed=conv_layers_fixed)
    if not is_fine_tuning and not is_undistorted_output_precomputed:
        baseline_model, _ = initialize_model(Model_mode(model_mode).name, True, requires_grad=False)
    
    #Run on multiple GPUs
    model = nn.DataParallel(model).to(device)
    if not is_fine_tuning and not is_undistorted_output_precomputed:
        baseline_model = nn.DataParallel(baseline_model).to(device)
        baseline_model.eval()
    
    if init_weights is not None:
        print("Loading provided initialization weights...")
        model.load_state_dict(torch.load(init_weights))

    bit_per_pixel_scale = 8/float((input_size)**2)

    params_to_update = model.parameters()
    if conv_layers_fixed:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
    
    optimizer = torch.optim.Adam(params_to_update, betas=(0.5, 0.99), lr=learning_rate)
    
    #Initialize loss
    classification_loss_fn = nn.CrossEntropyLoss()
    if not is_fine_tuning:
        expert_loss_fn = nn.KLDivLoss(reduction='sum')

    datasets, dataloaders, folder_string = initialize_datasets_dataloader(distortion, input_size, distortion_precompute_mode, batch_size, num_workers) #Initialize datasets and dataloaders    
    lowest_mean_val_loss_obtained, lowest_mean_val_loss_not_beaten_counter = 727, 0 #Used to reduce learning rate if validation loss do not improve anymore
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        for phase in ['train', 'val']:
            is_undistorted_output_precomputed = os.path.exists(undistorted_output_path + phase + "/")

            if epoch == start_epoch and phase == 'train':
                os.makedirs("/weights/" + folder_string, exist_ok=True)
                os.makedirs("/csv/" + folder_string, exist_ok=True)
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
            accumulated_classification_loss, accumulated_expert_loss, accumulated_final_loss = 0.0, 0.0, 0.0
            imagenet_correctly_classified_samples_top_1, imagenet_correctly_classified_samples_top_5 = 0, 0
            cumulated_compressed_imgs_size, compressed_imgs_total_samples = 0, 0
            
            for idxs, undistorted_batch, distorted_batch, labels_batch, bitstreams_length in dataloaders[phase]:
                ###############################################################################################################
                """
                #Show images in batch for debug purpose
                mean_norm, std_norm = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32), torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
                reverse_trsfms = transforms.Compose([transforms.Normalize(mean=(-mean_norm / std_norm).tolist(), std=(1.0 / std_norm).tolist()), transforms.ToPILImage()]) #For debug purpose
                import matplotlib.pyplot as plt #For debug purpose
                for elem in range(len(undistorted_batch)):
                    undistorted_img = reverse_trsfms(undistorted_batch[elem])
                    distorted_img = reverse_trsfms(distorted_batch[elem])
                    plt.imshow(undistorted_img, interpolation=None)
                    plt.show(block=True)
                    plt.imshow(distorted_img, interpolation=None)
                    plt.show(block=True)
                """
                ###############################################################################################################

                if batch_i % (5*print_density) == 0:
                    print("Epoch " + str(epoch), " / ", str(num_epochs), "    |   ", "Batch " + str(batch_i), " / ", str(len(dataloaders[phase])))
                batch_i += 1

                if not is_fine_tuning:
                    undistorted_batch = undistorted_batch.to(device)
                distorted_batch, labels_batch, bitstreams_length = distorted_batch.to(device), labels_batch.to(device), bitstreams_length.to(device)

                ###############################################################################################################
                ##########          Compute classification and expert loss          ###########################################
                ###############################################################################################################
                
                optimizer.zero_grad()
                if not is_fine_tuning:
                    if not is_undistorted_output_precomputed: #Prediction not saved on disk, precomputing them... The goal is to avoid procomputing predictions on undistorted image at every epoch with baseline_model since it will always be the same
                        with torch.set_grad_enabled(False):
                            undistorted_output = baseline_model(undistorted_batch)

                            for i in range(len(idxs)):
                                base_img_path = datasets[phase].imgs[idxs[i]][0]
                                current_undistorted_output_path = undistorted_output_path + base_img_path[base_img_path.index(phase + "/n"):base_img_path.rindex(".")] + ".pt"
                                    
                                #print("base_img_path", base_img_path)
                                #print("current_undistorted_output_path", current_undistorted_output_path)
                                os.makedirs(os.path.dirname(current_undistorted_output_path), exist_ok=True)
                                torch.save(undistorted_output[i].clone(), current_undistorted_output_path)
                    else: #Prediction saved on disk, loading them...                        
                        undistorted_output = torch.stack([torch.load(undistorted_output_path + datasets[phase].imgs[idx][0][datasets[phase].imgs[idx][0].index(phase + "/n"):datasets[phase].imgs[idx][0].rindex(".")] + ".pt") for idx in idxs])
                    
                with torch.set_grad_enabled(phase == 'train'):
                    distorted_output = model(distorted_batch)
                
                    classification_loss = classification_loss_fn(distorted_output, labels_batch)

                    if not is_fine_tuning:
                        expert_loss = alpha*expert_loss_fn(nn.functional.log_softmax(distorted_output, dim=1), nn.functional.softmax(undistorted_output, dim=1))
                    
                    if is_fine_tuning:
                        final_loss = classification_loss
                    else:
                        final_loss = classification_loss + expert_loss

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
                    if not is_fine_tuning:
                        accumulated_expert_loss += expert_loss.item() * len(labels_batch)
                    accumulated_final_loss += final_loss.item() * len(labels_batch)
                    
                    cumulated_compressed_imgs_size += torch.sum(bitstreams_length).item()
                    batch_compressed_samples = bitstreams_length!=0
                    compressed_imgs_total_samples += torch.sum(batch_compressed_samples.int()).item()
                    
                    #We count only well classifed samples that were compressed (p!=1.0 in dataset can return sample that were not compressed)
                    imagenet_correctly_classified_samples_top_1  += torch.logical_and(top_1_accuracies,  batch_compressed_samples.int()).sum().item()
                    imagenet_correctly_classified_samples_top_5  += torch.logical_and(top_5_accuracies,  batch_compressed_samples.int()).sum().item()

                    if batch_i % print_density == 0:
                        epoch_batches_left = len(dataloaders[phase]) - (batch_i + 1)
                        time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                        print(f"    ---- ETA {time_left}")

            ###############################################################################################################
            ##########          END OF EPOCH          #####################################################################
            ###############################################################################################################
            
            accumulated_classification_loss = accumulated_classification_loss / (epsilon + len(datasets[phase]))
            if not is_fine_tuning:
                accumulated_expert_loss = accumulated_expert_loss / (epsilon + len(datasets[phase]))
            accumulated_final_loss = accumulated_final_loss / (epsilon + len(datasets[phase]))

            mean_image_size = cumulated_compressed_imgs_size / (epsilon + float(compressed_imgs_total_samples))
            mean_bit_pet_pixel = bit_per_pixel_scale*cumulated_compressed_imgs_size / (epsilon + float(compressed_imgs_total_samples))
            
            if mean_image_size <= 0: #Invalid rate, load rate from /data/_stat_compressed/ for current distortion if available
                mean_image_size_tmp, mean_bit_pet_pixel_tmp = get_distortion_mean_rate(distortion, phase)
                if not mean_image_size_tmp == -1: #Successfully found rate for current distortion, updating mean_image_size and mean_bit_pet_pixel
                    mean_image_size, mean_bit_pet_pixel = mean_image_size_tmp, mean_bit_pet_pixel_tmp

            top_1_compressed_accuracy =  imagenet_correctly_classified_samples_top_1  / (epsilon + float(compressed_imgs_total_samples))
            top_5_compressed_accuracy =  imagenet_correctly_classified_samples_top_5  / (epsilon + float(compressed_imgs_total_samples))
            
            #Printing logs at every train/val phase, at every epoch
            table =  [["Metrics (" + ("TRAINING" if phase == 'train' else "VALIDATION") + " set)", "values"]]
            table += [["learning rate", "%.8f" % learning_rate]]
            table += [["classification loss", "%.8f" % accumulated_classification_loss]]
            if not is_fine_tuning:
                table += [["expert loss", "%.8f" % accumulated_expert_loss]]
            table += [["final loss", "%.8f" % accumulated_final_loss]]
            table += [["mean image size", "%.4f" % mean_image_size]]
            table += [["mean bit pet pixel", "%.8f" % mean_bit_pet_pixel]]
            table += [["distortion", distortion.DISTORTION_STRING]]
            table += [["Quality", distortion.quality]]
            if distortion.color_subsampling is not None:
                table += [["Color subsampling", distortion.color_subsampling]]
            table += [["top-1 compressed accuracy",  "%.8f" % top_1_compressed_accuracy]]
            table += [["top-5 compressed accuracy",  "%.8f" % top_5_compressed_accuracy]]
            print(AsciiTable(table).table)

            #Saving csv files containing various informations at every train/val phase, at every epoch
            csv_file = open("/csv/" + folder_string + "classification_loss__" + phase + "_loss__epoch_" + str(epoch) + ".csv", "w")
            csv_file.write(str(epoch) + "\n" + str(accumulated_classification_loss))
            csv_file.close()
            if not is_fine_tuning:
                csv_file = open("/csv/" + folder_string + "expert_loss__" + phase + "_loss__epoch_" + str(epoch) + ".csv", "w")
                csv_file.write(str(epoch) + "\n" + str(accumulated_expert_loss))
                csv_file.close()
            csv_file = open("/csv/" + folder_string + "final_loss__" + phase + "_loss__epoch_" + str(epoch) + ".csv", "w")
            csv_file.write(str(epoch) + "\n" + str(accumulated_final_loss))
            csv_file.close()
            csv_file = open("/csv/" + folder_string + "Accuracy_top_1__" + phase + "_accuracy__epoch_" + str(epoch) + ".csv", "w")
            csv_file.write(str(epoch) + "\n" + str(100*top_1_compressed_accuracy))
            csv_file.close()
            csv_file = open("/csv/" + folder_string + "Accuracy_top_5__" + phase + "_accuracy__epoch_" + str(epoch) + ".csv", "w")
            csv_file.write(str(epoch) + "\n" + str(100*top_5_compressed_accuracy))
            csv_file.close()

            csv_file = open("/csv/" + folder_string + "Rate_accuracy_top_1__" + phase + "_accuracy__epoch_" + str(epoch) + ".csv", "w")
            csv_file.write(str(mean_image_size) + "\n" + str(100*top_1_compressed_accuracy))
            csv_file.close()
            csv_file = open("/csv/" + folder_string + "Rate_accuracy_top_5__" + phase + "_accuracy__epoch_" + str(epoch) + ".csv", "w")
            csv_file.write(str(mean_image_size) + "\n" + str(100*top_5_compressed_accuracy))
            csv_file.close()

            csv_file = open("/csv/" + folder_string + "RD_curve_top_1__" + phase + "_accuracy__epoch_" + str(epoch) + ".csv", "w")
            csv_file.write(str(mean_bit_pet_pixel) + "\n" + str(100*top_1_compressed_accuracy))
            csv_file.close()
            csv_file = open("/csv/" + folder_string + "RD_curve_top_5__" + phase + "_accuracy__epoch_" + str(epoch) + ".csv", "w")
            csv_file.write(str(mean_bit_pet_pixel) + "\n" + str(100*top_5_compressed_accuracy))
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
                            if datasets is not None and distortion_precompute_mode == Distortion_precompute.PRECOMPUTED_ENCODING_STACKED:
                                for phase in ['train', 'val']:
                                    datasets[phase].remove_rec_yuv_files()
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
                torch.save(model.state_dict(), f"/weights/" + folder_string + "model_wts_epoch_" + str(epoch) + ".pth")
                
################################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--init", help="Weights to use at initialisation (by default, it is a pre-trained model on losslessly compressed images).", type=str, required=False)
    opt = parser.parse_args()
    init_weights = opt.init

    #True: Fine-tuning, False: Expert training
    fine_tuning     = True
    expert_training = False

    """    
    We recommand to use Distortion_precompute.PRECOMPUTED_ON_THE_FLY for JPEG and JPEG2000 distortions
    
    We highly recommand to use Distortion_precompute.PRECOMPUTED_ENCODING_DECODING for BPG, even though it requires a lot of disk space.
    A much slower alternative is to use Distortion_precompute.PRECOMPUTED_ENCODING. This setting is not recommanded because of the huge slow down at training time (requires to write temporary files for decoding).
    Note that, to use Distortion_precompute.PRECOMPUTED_ENCODING_DECODING for a given distortion, you first need to precompute the encoding of images in the whole dataset using script dataset_preprocessing.py with Distortion_precompute.PRECOMPUTED_ENCODING_DECODING.
    
    
    For each distortion, learning rate and alpha values are the ones that provided the best results among the one tested. These results are the one provided in the article.
    """

    """ Fine-tuning / ResNet50 """
    #main(Model_mode.resnet50, init_weights, Distortion(Distortion_mode.LOSSLESS                                     ), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       ,     fine_tuning, learning_rate = 1e-6, alpha = 1e-2)
    #main(Model_mode.resnet50, init_weights, Distortion(Distortion_mode.JPEG    , quality=75, color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       ,     fine_tuning, learning_rate = 5e-6, alpha = 5e-3)
    #main(Model_mode.resnet50, init_weights, Distortion(Distortion_mode.JPEG    , quality=10, color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       ,     fine_tuning, learning_rate = 1e-5, alpha = 1e-2)
    #main(Model_mode.resnet50, init_weights, Distortion(Distortion_mode.JPEG    , quality=5 , color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       ,     fine_tuning, learning_rate = 5e-6, alpha = 1e-2)
    #main(Model_mode.resnet50, init_weights, Distortion(Distortion_mode.JPEG2000, quality=75                         ), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       ,     fine_tuning, learning_rate = 5e-6, alpha = 1e-2)
    #main(Model_mode.resnet50, init_weights, Distortion(Distortion_mode.JPEG2000, quality=200                        ), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       ,     fine_tuning, learning_rate = 5e-6, alpha = 1e-2)
    #main(Model_mode.resnet50, init_weights, Distortion(Distortion_mode.JPEG2000, quality=400                        ), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       ,     fine_tuning, learning_rate = 5e-6, alpha = 1e-2)
    #main(Model_mode.resnet50, init_weights, Distortion(Distortion_mode.BPG     , quality=41, color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ENCODING_DECODING,     fine_tuning, learning_rate = 5e-6, alpha = 1e-2)
    #main(Model_mode.resnet50, init_weights, Distortion(Distortion_mode.BPG     , quality=46, color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ENCODING_DECODING,     fine_tuning, learning_rate = 5e-6, alpha = 1e-2)
    #main(Model_mode.resnet50, init_weights, Distortion(Distortion_mode.BPG     , quality=51, color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ENCODING_DECODING,     fine_tuning, learning_rate = 5e-6, alpha = 1e-2)

    """ Expert training / ResNet50 """
    #main(Model_mode.resnet50, init_weights, Distortion(Distortion_mode.LOSSLESS                                     ), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       , expert_training, learning_rate = 1e-6, alpha = 1e-3)
    #main(Model_mode.resnet50, init_weights, Distortion(Distortion_mode.JPEG    , quality=75, color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       , expert_training, learning_rate = 5e-6, alpha = 5e-3)
    main(Model_mode.resnet50, init_weights, Distortion(Distortion_mode.JPEG    , quality=10, color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       , expert_training, learning_rate = 1e-5, alpha = 1e-2)
    #main(Model_mode.resnet50, init_weights, Distortion(Distortion_mode.JPEG    , quality=5 , color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       , expert_training, learning_rate = 5e-6, alpha = 1e-2)
    #main(Model_mode.resnet50, init_weights, Distortion(Distortion_mode.JPEG2000, quality=75                         ), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       , expert_training, learning_rate = 5e-6, alpha = 1e-2)
    #main(Model_mode.resnet50, init_weights, Distortion(Distortion_mode.JPEG2000, quality=200                        ), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       , expert_training, learning_rate = 1e-5, alpha = 1e-2)
    #main(Model_mode.resnet50, init_weights, Distortion(Distortion_mode.JPEG2000, quality=400                        ), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       , expert_training, learning_rate = 1e-5, alpha = 1e-2)
    #main(Model_mode.resnet50, init_weights, Distortion(Distortion_mode.BPG     , quality=41, color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ENCODING_DECODING, expert_training, learning_rate = 5e-6, alpha = 1e-2)
    #main(Model_mode.resnet50, init_weights, Distortion(Distortion_mode.BPG     , quality=46, color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ENCODING_DECODING, expert_training, learning_rate = 5e-6, alpha = 1e-2)
    #main(Model_mode.resnet50, init_weights, Distortion(Distortion_mode.BPG     , quality=51, color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ENCODING_DECODING, expert_training, learning_rate = 5e-6, alpha = 1e-2)

    """ Fine-tuning / MnasNet """
    #main(Model_mode.mnasnet, init_weights, Distortion(Distortion_mode.LOSSLESS                                     ), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       ,     fine_tuning, learning_rate = 1e-5, alpha = 1e-2)
    #main(Model_mode.mnasnet, init_weights, Distortion(Distortion_mode.JPEG    , quality=75, color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       ,     fine_tuning, learning_rate = 1e-5, alpha = 1e-2)
    #main(Model_mode.mnasnet, init_weights, Distortion(Distortion_mode.JPEG    , quality=10, color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       ,     fine_tuning, learning_rate = 1e-5, alpha = 1e-2)
    #main(Model_mode.mnasnet, init_weights, Distortion(Distortion_mode.JPEG    , quality=5 , color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       ,     fine_tuning, learning_rate = 1e-5, alpha = 1e-2)
    #main(Model_mode.mnasnet, init_weights, Distortion(Distortion_mode.JPEG2000, quality=75                         ), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       ,     fine_tuning, learning_rate = 1e-5, alpha = 1e-2)
    #main(Model_mode.mnasnet, init_weights, Distortion(Distortion_mode.JPEG2000, quality=200                        ), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       ,     fine_tuning, learning_rate = 1e-5, alpha = 1e-2)
    #main(Model_mode.mnasnet, init_weights, Distortion(Distortion_mode.JPEG2000, quality=400                        ), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       ,     fine_tuning, learning_rate = 1e-5, alpha = 1e-2)
    #main(Model_mode.mnasnet, init_weights, Distortion(Distortion_mode.BPG     , quality=41, color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ENCODING_DECODING,     fine_tuning, learning_rate = 1e-5, alpha = 1e-2)
    #main(Model_mode.mnasnet, init_weights, Distortion(Distortion_mode.BPG     , quality=46, color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ENCODING_DECODING,     fine_tuning, learning_rate = 1e-5, alpha = 1e-2)
    #main(Model_mode.mnasnet, init_weights, Distortion(Distortion_mode.BPG     , quality=51, color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ENCODING_DECODING,     fine_tuning, learning_rate = 1e-5, alpha = 1e-2)

    """ Expert training / MnasNet """
    #main(Model_mode.mnasnet, init_weights, Distortion(Distortion_mode.LOSSLESS                                     ), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       , expert_training, learning_rate = 1e-5, alpha = 1e-2)
    #main(Model_mode.mnasnet, init_weights, Distortion(Distortion_mode.JPEG    , quality=75, color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       , expert_training, learning_rate = 1e-5, alpha = 1e-2)
    #main(Model_mode.mnasnet, init_weights, Distortion(Distortion_mode.JPEG    , quality=10, color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       , expert_training, learning_rate = 1e-5, alpha = 1e-2)
    #main(Model_mode.mnasnet, init_weights, Distortion(Distortion_mode.JPEG    , quality=5 , color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       , expert_training, learning_rate = 1e-5, alpha = 1e-2)
    #main(Model_mode.mnasnet, init_weights, Distortion(Distortion_mode.JPEG2000, quality=75                         ), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       , expert_training, learning_rate = 1e-5, alpha = 1e-2)
    #main(Model_mode.mnasnet, init_weights, Distortion(Distortion_mode.JPEG2000, quality=200                        ), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       , expert_training, learning_rate = 1e-5, alpha = 1e-2)
    #main(Model_mode.mnasnet, init_weights, Distortion(Distortion_mode.JPEG2000, quality=400                        ), Distortion_precompute.PRECOMPUTED_ON_THE_FLY       , expert_training, learning_rate = 1e-5, alpha = 1e-2)
    #main(Model_mode.mnasnet, init_weights, Distortion(Distortion_mode.BPG     , quality=41, color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ENCODING_DECODING, expert_training, learning_rate = 1e-5, alpha = 1e-2)
    #main(Model_mode.mnasnet, init_weights, Distortion(Distortion_mode.BPG     , quality=46, color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ENCODING_DECODING, expert_training, learning_rate = 1e-5, alpha = 1e-2)
    #main(Model_mode.mnasnet, init_weights, Distortion(Distortion_mode.BPG     , quality=51, color_subsampling="420"), Distortion_precompute.PRECOMPUTED_ENCODING_DECODING, expert_training, learning_rate = 1e-5, alpha = 1e-2)