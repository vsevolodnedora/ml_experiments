import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoder of the cVAE
    """

    def __init__(self, image_size=150, hidden_dim=50, z_dim=10, class_size=7):
        """
        :param image_size: Size of 1D "images" of data set i.e. spectrum size
        :param hidden_dim: Dimension of hidden layer
        :param z_dim: Dimension of latent space
        :param class_size: Dimension of conditioning variables

        """
        super().__init__()

        # nn.Linear(latent_dims, 512)
        self.layers_mu = nn.Sequential(
            nn.Linear(image_size + class_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, z_dim),
        )

        self.layers_logvar = nn.Sequential(
            nn.Linear(image_size + class_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, x):
        """
        Compute single pass through the encoder

        :param x: Concatenated images and corresponding conditioning variables
        :return: Mean and log variance of the encoder's distribution
        """
        mean = self.layers_mu(x)
        logvar = self.layers_logvar(x)
        return mean, logvar
class Decoder(nn.Module):
    """
    Decoder of cVAE
    """

    def __init__(self, image_size=150, hidden_dim=50, z_dim=10, c=7):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(z_dim + c, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, image_size),
            nn.Sigmoid(),
        )

    def forward(self, z):
        """
        Compute single pass through the decoder

        :param z: Concatenated sample of hidden variables and the originally inputted conditioning variables
        :return: Mean of decoder's distirbution
        """
        mean = self.layers(z)
        return mean
class CVAE_OLD(nn.Module):
    """
        Base pytorch cVAE class
    """
    name = "CVAE"
    def __init__(self, image_size=150, hidden_dim=50, z_dim=10, c=7, init_weights=True, **kwargs):
        """
        :param image_size: Size of 1D "images" of data set i.e. spectrum size
        :param hidden_dim: Dimension of hidden layer
        :param z_dim: Dimension of latent space (latent_units)
        :param c: Dimension of conditioning variables
        """
        super(CVAE_OLD, self).__init__()
        self.z_dim = z_dim # needed for inference
        # self.c = c
        self.image_size=image_size
        # self.hidden_dim=hidden_dim
        self.encoder = Encoder(image_size, hidden_dim, z_dim, c) # self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(image_size, hidden_dim, z_dim, c) # self.decoder = Decoder(latent_dims)

        if init_weights:
            self.init_weights()

        self.model_settings={
            "name":"Kamile_CVAE",
            "image_size":image_size,
            "hidden_dim":hidden_dim,
            "z_dim":z_dim,
            "c":c,
            "init_weights":init_weights
        }

    @classmethod
    def init_from_dict(cls, dict : dict):
        # for key in cls._parameter_constraints.keys():
        #     if not (key in dict.keys):
        #         raise KeyError(f"Cannot initialize model. Parameter {key} is missing")

        return cls(**dict)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def encode(self, x, c):
        pass

    def decode(self, z, c):
        pass

    def forward(self, x, c):
        """
        Compute one single pass through decoder and encoder

        :param c: Conditioning variables corresponding to images/spectra
        :param x: Images/spectra
        :return: Mean returned by decoder, mean returned by encoder, log variance returned by encoder
        """

        # print("1 x={} y={}".format(x.shape, y.shape))
        x = torch.cat((x.view(-1, self.image_size), c), dim=1)
        mu, logvar = self.encoder(x) # # Q(z|x, c)
        # print(f"2 y={y.shape}")
        # re-parametrize
        z = self.reparameterize(mu, logvar)
        # print(f"3 sample={sample.shape}")
        z = torch.cat((z, c), dim=1)
        # print(f"4 cat(sample,x) -> z={z.shape}")
        mean_dec = self.decoder(z)
        # print(f"5 mean_dec={mean_dec.shape}")
        return (mean_dec, mu, logvar, z)

    def init_weights(self):
        """
            Initialize weight of recurrent layers
        """
        for name, param in self.encoder.named_parameters():
            if 'bias' in name:
                nn.init.normal_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.decoder.named_parameters():
            if 'bias' in name:
                nn.init.normal_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

class CVAE_BASIC_OLD(nn.Module):
    name = "CVAE"
    def __init__(self, feature_size=8192, hidden_dim=400, latent_size=20, class_size=7, init_weights=True, **kwargs):
        # image_size = 150, hidden_dim = 50, z_dim = 10, c = 7, init_weights = True, ** kwargs
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size
        self.latent_size = latent_size

        # encode
        self.fc1  = nn.Linear(feature_size + class_size, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_size)
        self.fc22 = nn.Linear(hidden_dim, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + class_size, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

        if init_weights:
            self.init_weights()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        if torch.any(torch.isnan(z_mu)):
            raise ValueError()
        return (z_mu, z_var)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        h3 = self.elu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        if torch.any(torch.isnan(x)):
            raise ValueError()
        if torch.any(torch.isnan(c)):
            raise ValueError()
        mu, logvar = self.encode(x.view(-1, self.feature_size), c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        if torch.any(torch.isnan(x_recon)):
            raise ValueError()
        return (x_recon, mu, logvar)

    def init_weights(self):
        """
            Initialize weight of recurrent layers
        """
        for name, param in self.encoder.named_parameters():
            if 'bias' in name:
                nn.init.normal_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.decoder.named_parameters():
            if 'bias' in name:
                nn.init.normal_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)


class CVAE_(nn.Module):
    name = "CVAE"

    def __init__(self, feature_size=8192, hidden_dim=400,
                 latent_size=20, class_size=7, init_weights=True, **kwargs):
        # image_size = 150, hidden_dim = 50, z_dim = 10, c = 7, init_weights = True, ** kwargs
        super(CVAE_, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size
        self.latent_size = latent_size

        # encode
        # self.fc1 = nn.Linear(feature_size + class_size, hidden_dim)
        # self.fc21 = nn.Linear(hidden_dim, latent_size)
        # self.fc22 = nn.Linear(hidden_dim, latent_size)
        self.layers_mu = nn.Sequential(
            nn.Linear(in_features=feature_size + class_size,
                      out_features=hidden_dim),
            nn.LeakyReLU(),# nn.Tanh(),
            nn.Linear(in_features=hidden_dim,
                      out_features=hidden_dim),
            nn.LeakyReLU(),#nn.Tanh(),
            nn.Linear(in_features=hidden_dim,
                      out_features=latent_size),
        )
        self.layers_logvar = nn.Sequential(
            nn.Linear(in_features=feature_size + class_size,
                      out_features=hidden_dim),
            nn.LeakyReLU(),#nn.Tanh(),
            nn.Linear(in_features=hidden_dim,
                      out_features=hidden_dim),
            nn.LeakyReLU(),#nn.Tanh(),
            nn.Linear(in_features=hidden_dim,
                      out_features=latent_size),
        )

        # decode
        # self.fc3 = nn.Linear(latent_size + class_size, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim, feature_size)
        self.layers = nn.Sequential(
            nn.Linear(in_features=latent_size + class_size,
                      out_features=hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim,
                      out_features=hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim,
                      out_features=feature_size),
            nn.Sigmoid(),
        )

        # self.elu = nn.ELU()
        # self.sigmoid = nn.Sigmoid()

        if init_weights:
            self.init_weights()

    def encode(self, x, c):  # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([x, c], 1)  # (bs, feature_size+class_size)
        # h1 = self.elu(self.fc1(inputs))
        # z_mu = self.fc21(h1)
        # z_var = self.fc22(h1)
        # if torch.any(torch.isnan(z_mu)):
        #     raise ValueError()
        z_mu = self.layers_mu(inputs)
        z_var = self.layers_logvar(inputs)
        return (z_mu, z_var)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):  # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([z, c], 1)  # (bs, latent_size+class_size)
        # h3 = self.elu(self.fc3(inputs))
        # return self.sigmoid(self.fc4(h3))
        x_reconstruct = self.layers(inputs)
        return x_reconstruct

    def forward(self, x, c):
        if torch.any(torch.isnan(x)):
            raise ValueError()
        if torch.any(torch.isnan(c)):
            raise ValueError()
        mu, logvar = self.encode(x.view(-1, self.feature_size), c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        if torch.any(torch.isnan(x_recon)):
            raise ValueError()
        return (x_recon, mu, logvar)

    def init_weights(self):
        """
            Initialize weight of recurrent layers
        """
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.normal_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        # for name, param in self.named_parameters():
        #     if 'bias' in name:
        #         nn.init.normal_(param, 0.0)
        #     elif 'weight' in name:
        #         nn.init.xavier_normal_(param)

class CVAE(nn.Module):
    name = "CVAE"

    def __init__(self, feature_size=8192, hidden_dim=400,
                 latent_size=20, class_size=7, init_weights=True, **kwargs):
        # image_size = 150, hidden_dim = 50, z_dim = 10, c = 7, init_weights = True, ** kwargs
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size
        self.latent_size = latent_size

        # encode
        # self.fc1 = nn.Linear(feature_size + class_size, hidden_dim)
        # self.fc21 = nn.Linear(hidden_dim, latent_size)
        # self.fc22 = nn.Linear(hidden_dim, latent_size)
        self.layers_mu = nn.Sequential(
            # nn.Linear(in_features=feature_size + class_size,
            #           out_features=hidden_dim),
            # nn.LeakyReLU(),# nn.Tanh(),
            # nn.Linear(in_features=hidden_dim,
            #           out_features=hidden_dim),
            # nn.LeakyReLU(),#nn.Tanh(),
            # nn.Linear(in_features=hidden_dim,
            #           out_features=latent_size),
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=0),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.Linear(in_features=feature_size,
                      out_features=latent_size)
        )
        self.layers_logvar = nn.Sequential(
            nn.Linear(in_features=feature_size + class_size,
                      out_features=hidden_dim),
            nn.LeakyReLU(),#nn.Tanh(),
            nn.Linear(in_features=hidden_dim,
                      out_features=hidden_dim),
            nn.LeakyReLU(),#nn.Tanh(),
            nn.Linear(in_features=hidden_dim,
                      out_features=latent_size),
        )

        # decode
        # self.fc3 = nn.Linear(latent_size + class_size, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim, feature_size)
        self.layers = nn.Sequential(
            nn.Linear(in_features=latent_size + class_size,
                      out_features=hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim,
                      out_features=hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim,
                      out_features=feature_size),
            nn.Sigmoid(),
        )

        # self.elu = nn.ELU()
        # self.sigmoid = nn.Sigmoid()

        if init_weights:
            self.init_weights()

    def encode(self, x, c):  # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([x, c], 1)  # (bs, feature_size+class_size)
        # h1 = self.elu(self.fc1(inputs))
        # z_mu = self.fc21(h1)
        # z_var = self.fc22(h1)
        # if torch.any(torch.isnan(z_mu)):
        #     raise ValueError()
        z_mu = self.layers_mu(inputs)
        z_var = self.layers_logvar(inputs)
        return (z_mu, z_var)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):  # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([z, c], 1)  # (bs, latent_size+class_size)
        # h3 = self.elu(self.fc3(inputs))
        # return self.sigmoid(self.fc4(h3))
        x_reconstruct = self.layers(inputs)
        return x_reconstruct

    def forward(self, x, c):
        if torch.any(torch.isnan(x)):
            raise ValueError()
        if torch.any(torch.isnan(c)):
            raise ValueError()
        mu, logvar = self.encode(x.view(-1, self.feature_size), c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        if torch.any(torch.isnan(x_recon)):
            raise ValueError()
        return (x_recon, mu, logvar)

    def init_weights(self):
        """
            Initialize weight of recurrent layers
        """
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.normal_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        # for name, param in self.named_parameters():
        #     if 'bias' in name:
        #         nn.init.normal_(param, 0.0)
        #     elif 'weight' in name:
        #         nn.init.xavier_normal_(param)

def load_model(model_dir, model_metada, device):
    """ loads the model (eval mode) with all dictionaries """
    model = CVAE.init_from_dict(model_metada)
    state = torch.load(model_dir+"model.pt", map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    model.to(device)
    return (model, state)
