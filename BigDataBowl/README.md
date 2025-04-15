This is our approach to the Big Data Bowl 2025. We had to use pre-snap data to predict post-snap movement. 

We had decided to work on using pre-snap movement to predict whether the play was a run or pass play. 

We used a deep learning architecture consisting of an CNN backbone. The output of this was concatenated with the processed persistent homology data (see PHG_Net for detailed understanding), and with situational data, and was passed through an MLP. 

We achieved an accuracy of approximately 70%. 
