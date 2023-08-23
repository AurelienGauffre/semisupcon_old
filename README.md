To do/test :
- [x] Pseudo-Labels Accuracy display (as in simmatch)
- [x] Use pretrained weight for wideresnet (
- [ ] checks that these pretrained works are loaded correctly : why the loss is not continous after loader
- [ ] ajouter le fait qu'on puisse choisir un facteur pour 
- [ ] pourquoi une plus grande BS overfit plus ? Faire un truc pour la bBS
- [ ] different threshold for pseudo-labels (one for ce that trains the logtis head, one for supcon loss)
prendre moins en compte la partie non supervisée de la loss supcon
- Loss supcon unifiée
- Pour la CE pseudolabel, rajouter un deuxime threshold plus haut ! Et regarder le % de pseudo labelisé

- [ ] question : la partie x_ulb_w est detachée, pour calculer la partie ce_unsuper 
(qui entraine la projection lineaire), il faudrait en fait attacher le gradient )
D; ailleur si c'est vriament détahé le fait de rajouter la loss ce_unsupervised ne devrait rien changer
- [ ] adaptation a flexmatch ?
- [ ] adaptation to simsiam : for the pretraining could work and what about supcon loss ? Does it becomes exactly consistency reg ?
- [ ] multi crop ?
- [ ] cutmix ?

- essayer plus petie BS
- transfer learning apres preentrainement contrastif
- dropout ?