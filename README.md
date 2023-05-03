To do/test :
-[ ] Use pretrained weight for wideresnet ?
-[ ] dans net/wideresent remplacer self.classifier = nn.Linear(channels[3], num_classes)
par un MLP : idée est de mettre moins de pression sur
- [ ] ajouter le fait qu'on puisse choisir un facteur pour 
prendre moins en compte la partie non supervisée de la loss supcon



I am using supcon loss (supervised contrastive learning) that takes simply two arguments : embeddings (BxD), labels (B). However I am in a semi-supervised setting. I have some embedings feats_x_lb for which I have labels y_lb. But I also have embeddings that come from unsupervised data
feats_x_ulb, pseudo labels y_pseudo and a binary mask mask which says which unsupervised embedings and pseudo labels I want to add to my loss. How to make a pytorch code to make only one call to supcon loss with all my labels (supervised and unsupervised for which the mask is 1) in an effective manner ?