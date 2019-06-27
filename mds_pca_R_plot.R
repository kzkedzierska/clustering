library(tidyverse)
library(gridExtra)

mds <- iris[,1:4] %>% 
  dist() %>% 
  cmdscale() %>% 
  as.data.frame() %>% 
  mutate(method = "mds", 
         label = iris$Species) %>% 
  rename(x = V1, y = V2)

pca <- (iris[,1:4] %>% prcomp())$x %>% 
  as.data.frame() %>%
  mutate(method = "pca", 
         label = iris$Species)

p1 <- pca %>% 
  ggplot(., aes(PC1, PC2, color = label)) + 
  geom_point() + 
  scale_color_viridis_d() + 
  labs(title = "PCA")

p2 <- mds %>% 
  ggplot(., aes(x, y, color = label)) + 
  geom_point() +
  scale_color_viridis_d() + 
  labs(title = "MDS")

png("./mds_pca_R_plot.png")
grid.arrange(p1, p2)
dev.off()

write_tsv(iris, "./iris.tsv")
