import matplotlib.pyplot as plt




def plotter(model_name, train_losses, train_pos_acc, train_neg_acc,
            val_losses, val_pos_acc, val_neg_acc, loss, loss_type='Weighted BCE Loss'):
    plt.subplot(1,2,1)   
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(loss_type)
    plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(train_pos_acc, label='train_pos')
    plt.plot(train_neg_acc, label='train_neg')
    plt.plot(val_pos_acc, label='val_pos')
    plt.plot(val_neg_acc, label='val_neg')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.suptitle(model_name)
    plt.show()
    return

def show_prog(epoch, train_loss, val_loss, train_pos_acc, train_neg_acc, val_pos_acc, val_neg_acc, time_elapsed):
    print('Epoch', epoch+1, ': train avg los:', round(train_loss, 3), 'validation avg los:', round(val_loss, 3))
    print('Epoch', epoch+1, ': train pos acc:', round(train_pos_acc, 3), 'validation pos acc:', round(val_pos_acc, 3))
    print('Epoch', epoch+1, ': train neg acc:', round(train_neg_acc, 3), 'validation neg acc:', round(val_neg_acc, 3))
    print('total runtime:', str(round(time_elapsed, 2)))