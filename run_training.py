from vcm.training import train

# input / output data
csv_file = 'C:/mlvq_fhl/cmos_fullset.csv'
data_dir = 'C:/mlvq_fhlclip_with_cmos'
output_dir = 'C:/mlvq_fhl/output'

# training settings
#epochs = 1000
epochs = 5
features = ['integer_motion','integer_motion2', 'integer_adm2', 'integer_adm_scale0', 'integer_adm_scale1', 'integer_adm_scale2', 
            'integer_adm_scale3', 'integer_vif_scale0', 'integer_vif_scale1', 'integer_vif_scale2', 'integer_vif_scale3']
hidden_size = 256
num_layers = 6
bs = 128
lr = 1e-4
lr_patience = 20

if __name__ == "__main__":
    train(csv_file, data_dir, output_dir, epochs, features, hidden_size, num_layers, bs, lr, lr_patience)