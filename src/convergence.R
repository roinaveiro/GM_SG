library(ggplot2)
final_red = read.csv("convergence.csv")
p = ggplot(final_red, aes(x= Epoch, y = Loss)) + theme_bw()
p = p + geom_line(aes(color=factor(Experiment)), size=0.8)
p = p + theme(plot.title = element_text(hjust = 0.5))
p = p + ggtitle("Convergence for several starting points")
p = p +scale_color_manual(name = "Loss",
                          values = replicate(20, "grey41")) 
p = p + guides(color=F)
p

ggsave(filename = "survival-curves.eps",
       plot = print(p), device = "eps")

