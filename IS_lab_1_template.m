% Classification using perceptron
clc
clear all
close all
% Reading apple images
A1=imread('apple_04.jpg');
A2=imread('apple_05.jpg');
A3=imread('apple_06.jpg');
A4=imread('apple_07.jpg');
A5=imread('apple_11.jpg');
A6=imread('apple_12.jpg');
A7=imread('apple_13.jpg');
A8=imread('apple_17.jpg');
A9=imread('apple_19.jpg');

% Reading pears images
P1=imread('pear_01.jpg');
P2=imread('pear_02.jpg');
P3=imread('pear_03.jpg');
P4=imread('pear_09.jpg');

% Calculate for each image, colour and roundness
% For Apples
% 1st apple image(A1)
hsv_value_A1=spalva_color(A1); %color
metric_A1=apvalumas_roundness(A1); %roundness
% 2nd apple image(A2)
hsv_value_A2=spalva_color(A2); %color
metric_A2=apvalumas_roundness(A2); %roundness
% 3rd apple image(A3)
hsv_value_A3=spalva_color(A3); %color
metric_A3=apvalumas_roundness(A3); %roundness
% 4th apple image(A4)
hsv_value_A4=spalva_color(A4); %color
metric_A4=apvalumas_roundness(A4); %roundness
% 5th apple image(A5)
hsv_value_A5=spalva_color(A5); %color
metric_A5=apvalumas_roundness(A5); %roundness
% 6th apple image(A6)
hsv_value_A6=spalva_color(A6); %color
metric_A6=apvalumas_roundness(A6); %roundness
% 7th apple image(A7)
hsv_value_A7=spalva_color(A7); %color
metric_A7=apvalumas_roundness(A7); %roundness
% 8th apple image(A8)
hsv_value_A8=spalva_color(A8); %color
metric_A8=apvalumas_roundness(A8); %roundness
% 9th apple image(A9)
hsv_value_A9=spalva_color(A9); %color
metric_A9=apvalumas_roundness(A9); %roundness

%For Pears
%1st pear image(P1)
hsv_value_P1=spalva_color(P1); %color
metric_P1=apvalumas_roundness(P1); %roundness
%2nd pear image(P2)
hsv_value_P2=spalva_color(P2); %color
metric_P2=apvalumas_roundness(P2); %roundness
%3rd pear image(P3)
hsv_value_P3=spalva_color(P3); %color
metric_P3=apvalumas_roundness(P3); %roundness
%2nd pear image(P4)
hsv_value_P4=spalva_color(P4); %color
metric_P4=apvalumas_roundness(P4); %roundness

%selecting features(color, roundness, 3 apples and 2 pears)
%A1,A2,A3,P1,P2
%building matrix 2x5
x1=[hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_P1 hsv_value_P2];
x2=[metric_A1 metric_A2 metric_A3 metric_P1 metric_P2];
% estimated features are stored in matrix P:
P=[x1;x2];

%Desired output vector
T=[1;1;1;-1;-1]; % <- ČIA ANKSČIAU BUVO KLAIDA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

%% train single perceptron with two inputs and one output

%Set training parameters
w1 = randn(1); %weight of feature 1
w2 = randn(1); %weight of feature 2
b = randn(1); %bias
eta = 0.1 %training rate
e=0 %setting initial error to 0
max_epochs = 1000; %maximum number of training "rounds" to be performed
epoch = 0; %initial training round

%calculate the initial total error for our set of training images
for i=1:5 %for each image
    v = x1(i)*w1+x2(i)*w2+b; %calculate the weighted sum
    if v > 0 %calculate output
        y = 1;
    else
        y=-1;
    end
    e_total = e_total + (T(i) - y); %updates total error
end


%Training algorithm
while e_total ~= 0 && epoch < max_epochs % executes while the total error is not 0
    e_total = 0; %resets total error to 0
    for i=1:5
        v = x1(i)*w1+x2(i)*w2+b;
        if v > 0
            y = 1;
        else
            y=-1;
        end
        e = T(i) - y %tests if the output corresponds to the expected output
        w1 = w1 + eta*e*x1(i);
        w2 = w2 + eta*e*x2(i); 
        b = b + eta*e;
        e_total = e_total + abs(e);%calculate absolute total error
    end
    epoch = epoch+1; 
    fprintf('Epoch %d, Total Error: %d\n', epoch, e_total);
    %if the absolute total error of all the training examples is not 0,
    %loops with updated weights 
end
disp(['Trained weights: w1 = ', num2str(w1), ', w2 = ', num2str(w2)]);
disp(['Trained bias: b = ', num2str(b)]);

%Test for new images
x1=[hsv_value_A4 hsv_value_A5 hsv_value_A6 hsv_value_P3 hsv_value_P4];
x2=[metric_A4 metric_A5 metric_A6 metric_P3 metric_P4];
T=[1;1;1;-1;-1]; %updated expectations
accurate_classifications = 0
for i = 1:5
    v = x1(i) * w1 + x2(i) * w2 + b;
    if v > 0
        y = 1;
    else
        y = -1;
    end
    if y == T(i)
        accurate_classifications = accurate_classifications + 1;
    end
    
    fprintf('Example %d: Desired = %d, Output = %d\n', i, T(i), y);
end

% Calculate and display accuracy
accuracy = (accurate_classifications / 5) * 100;
fprintf('Training Accuracy: %.2f%%\n', accuracy);


% %%ADDITIONAL TASK - NAIVE BAYES CLASSIFIER%%
% 
% %We want to organize our data by class, we will use same training examples
% %as before 
% T = [1, 1, 1, -1, -1];  % 1 = Apple, -1 = Pear
% x1 = [hsv_value_A1, hsv_value_A2, hsv_value_A3, hsv_value_P1, hsv_value_P2];  % Color
% x2 = [metric_A1, metric_A2, metric_A3, metric_P1, metric_P2];  % Roundness
% 
% % Separate data by class
% apple_idx = find(T == 1); %Find function = Find elements in a matrix, we say matrix T and "1" elements here. It will return their place in the matrix
% pear_idx = find(T == -1);
% %It allows us to know where are pears and apples to then calculate their
% %variance and means
% 
% %Calculate mean and variance to then calculate Gaussian Likelihood
% x1_mean_apple=mean(x1(apple_idx))
% x1_mean_pear=mean(x1(pear_idx))
% 
% x2_mean_apple=mean(x2(apple_idx))
% x2_mean_pear=mean(x2(pear_idx))
% 
% x1_variance_apple=var(x1(apple_idx))
% x1_variance_pear=var(x1(apple_idx))
% 
% x2_variance_apple=var(x2(apple_idx))
% x2_variance_pear=var(x2(apple_idx))
% 
% %Calculate Gaussian Likelihood
% gaussian_likelihood = @(x, mean, var) (1 / sqrt(2 * pi * var)) * exp(-(x - mean)^2 / (2 * var));
