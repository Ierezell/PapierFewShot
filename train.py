
Network.encoder.load_state_dict(torch.load(PATH_WEIGHTS_ENCODER))
Network.decoder.load_state_dict(torch.load(PATH_WEIGHTS_DECODER))
Network.classifier.load_state_dict(torch.load(PATH_WEIGHTS_CLASSIFIER))

# # Plot some training images
print("Début de l'entrainement")
for i_epoch in range(NB_EPOCHS):
    for i_batch, batch in enumerate(train_loader):

        tic = time.time()

        images, labels = batch
        optimizer.zero_grad()

        anc, pos, neg = images

        lbl_anc, lbl_pos, lbl_neg = labels
        anc, pos, neg = anc.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
        lbl_anc, lbl_pos = lbl_anc.to(DEVICE), lbl_pos.to(DEVICE),
        lbl_neg = lbl_neg.to(DEVICE)

        sig_anc, im_anc, person_anc = Network(anc)
        sig_pos, im_pos, person_pos = Network(pos)
        sig_neg, im_neg, person_neg = Network(neg)

        loss_image_MSE = criterionMSE(anc, im_anc) +\
            criterionMSE(pos, im_pos) +\
            criterionMSE(neg, im_neg)
        loss_image_L1 = criterionL1(anc, im_anc) +\
            criterionL1(pos, im_pos) +\
            criterionL1(neg, im_neg)
        loss_image = ((loss_image_MSE +
                       loss_image_L1)/(IMAGE_SIZE*IMAGE_SIZE*BATCH_SIZE))/3

        loss_sig = (criterionTriplet(sig_anc, sig_pos, sig_neg)/(BATCH_SIZE))/3

        loss_pred = ((criterionRecogFaces(person_anc, lbl_anc) +
                      criterionRecogFaces(person_pos, lbl_pos) +
                      criterionRecogFaces(person_neg, lbl_neg))/(BATCH_SIZE))/3

        loss = loss_image + loss_sig + loss_pred

        loss_follow.append(loss)
        loss_image_follow.append(loss_image)
        loss_pred_follow.append(loss_pred)
        loss_sig_follow.append(loss_sig)
        losses_follow = [loss_follow,
                         loss_pred_follow, loss_sig_follow, loss_image_follow]

        loss.backward()
        optimizer.step()
        tac = time.time()
        recontructions = [im_anc, im_pos, im_neg]
        if i_batch % 300 == 0:
            visualize_info_batch(images, recontructions, tic, tac,
                                 len(train_datas), BATCH_SIZE,
                                 i_batch, i_epoch, losses_follow)

        if loss < best_loss and best_loss-loss > 0.2*best_loss:
            print('-'*10)
            print("Poids sauvegardés")
            print('-'*10)
            best_loss = loss
            # torch.save(Network.state_dict(), PATH_WEIGHTS)
            torch.save(Network.encoder.state_dict(), PATH_WEIGHTS_ENCODER)
            torch.save(Network.decoder.state_dict(), PATH_WEIGHTS_DECODER)
            torch.save(Network.classifier.state_dict(),
                       PATH_WEIGHTS_CLASSIFIER)
    clear_output(wait=True)
    sheduler.step()

torch.save(Network.state_dict(), ROOT_WEIGHTS+'/BACKUP_GOOD_LFWTRIPLET.pt')

torch.save(Network.encoder.state_dict(),
           ROOT_WEIGHTS+'/BACKUP_GOOD_LFWTRIPLETENCODER.pt')

torch.save(Network.decoder.state_dict(),
           ROOT_WEIGHTS+'/BACKUP_GOOD_LFWTRIPLETDECODER.pt')

torch.save(Network.classifier.state_dict(),
           './weights/BACKUP_GOOD_LFWTRIPLETCLASSIFIER.pt')

torch.save(Network.classifier.state_dict(), PATH_WEIGHTS_CLASSIFIER)
torch.save(Network.decoder.state_dict(), PATH_WEIGHTS_DECODER)
torch.save(Network.encoder.state_dict(), PATH_WEIGHTS_ENCODER)
