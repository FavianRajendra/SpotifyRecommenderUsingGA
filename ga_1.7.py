import streamlit as st
import numpy as np
import pandas as pd
from deap import base, creator, tools
import random
import time

# Load Spotify data
spotify_data = pd.read_csv('spotify_tracks_dataset.csv')

class SpotifyGeneticRecommender:
    def __init__(self, data: pd.DataFrame, playlist_length: int = 10):
        self.data = data
        self.playlist_length = playlist_length
        self.genres = sorted(data['track_genre'].unique())
        self.best_fitness_scores = []  # Initialize the best fitness scores list
        self._setup_deap()

    def _setup_deap(self):
        if 'FitnessMax' in creator.__dict__:
            del creator.FitnessMax
            del creator.Individual
            
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate_playlist)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _evaluate_playlist(self, individual, valid_indices):
        playlist = self.data.iloc[individual]
        popularity_score = playlist['popularity'].mean() / 100.0
        
        unique_artists = len(playlist['artists'].unique())
        diversity_score = unique_artists / len(individual)
        
        validity_score = sum(1 for idx in individual if idx in valid_indices) / len(individual)
        
        score = (0.4 * popularity_score + 0.3 * diversity_score + 0.3 * validity_score)
        return (score,)

    def _mutate_playlist(self, individual, indpb=0.1):
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = random.choice(self.valid_indices)
        return individual,

    def recommend(self, genre_preferences: list,
                population_size: int = 50,
                generations: int = 30,
                elitism_count: int = 1,
                convergence_threshold: float = 0.0003,
                max_convergence_iterations: int = 20,
                verbose: bool = False) -> pd.DataFrame:
        
        genre_mask = self.data['track_genre'].isin(genre_preferences)
        self.valid_indices = self.data[genre_mask].index.tolist()
        
        if not self.valid_indices:
            return pd.DataFrame()
            
        self.toolbox.register("track_idx", random.choice, self.valid_indices)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                            self.toolbox.track_idx, n=self.playlist_length)
        self.toolbox.register("population", tools.initRepeat, list,
                            self.toolbox.individual)
        
        self.toolbox.register("evaluate", self._evaluate_playlist, 
                            valid_indices=self.valid_indices)

        pop = self.toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)

        self.best_fitness_scores = []  # Initialize the list to store best fitness scores
        convergence_counter = 0

        for gen in range(generations):
            fitnesses = list(map(self.toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            hof.update(pop)

            best_fitness = hof[0].fitness.values[0]
            self.best_fitness_scores.append(best_fitness)  # Store in the class attribute

            if len(self.best_fitness_scores) > 1:
                if abs(self.best_fitness_scores[-1] - self.best_fitness_scores[-2]) < convergence_threshold:
                    convergence_counter += 1
                else:
                    convergence_counter = 0

                if convergence_counter >= max_convergence_iterations:
                    if verbose:
                        st.write(f"Convergence reached after {gen + 1} generations.")
                    break

            offspring = self.toolbox.select(pop, len(pop) - elitism_count)
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < 0.2:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            pop[:] = offspring + hof[:elitism_count]

        best_playlist = self.data.iloc[hof[0]][
            ['track_name', 'artists', 'track_genre', 'popularity']
        ].sort_values(by='popularity', ascending=False)

        fitness_df = pd.DataFrame({
            'Generation': range(1, len(self.best_fitness_scores) + 1),
            'Best Fitness Score': self.best_fitness_scores
        })

        st.write("Fitness scores for each generation:")
        st.dataframe(fitness_df)
        st.line_chart(fitness_df.set_index('Generation')['Best Fitness Score'])

        return best_playlist

# Initialize the recommender
recommender = SpotifyGeneticRecommender(spotify_data)

# Set page config
st.set_page_config(page_title="Spotify Playlist Magic Generator", page_icon="🎵", layout="wide")

# Add background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://path-to-your-music-themed-background-image.jpg');
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Animated introduction
def animated_text(text):
    placeholder = st.empty()
    for i in range(len(text) + 1):
        placeholder.markdown(f"## {text[:i]}_")
        time.sleep(0.05)
    placeholder.markdown(f"## {text}")

st.title("🎵 Spotify Playlist Magic Generator 🪄")
animated_text("Welcome to your personal playlist creator!")

# Create genre selection with colorful container
genre_container = st.container()
with genre_container:
    st.markdown("""
    <style>
    .genre-container {
        background-color: rgba(255, 105, 180, 0.3);
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="genre-container">', unsafe_allow_html=True)
    st.subheader("🎸 Choose Your Groove")
    available_genres = recommender.genres
    selected_genres = st.multiselect("Pick your favorite genres:", available_genres)
    st.markdown('</div>', unsafe_allow_html=True)

if st.button("🚀 Launch My Playlist"):
    if selected_genres:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate progress
        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text(f"Crafting your unique playlist... {i+1}%")
            time.sleep(0.05)  # Reduced sleep time for faster demo
        
        # Generate playlist
        playlist = recommender.recommend(selected_genres, verbose=True)

        if not playlist.empty:
            st.success("🎉 Your playlist is ready!")
            st.write(f"🎧 Your Personalized Playlist for {', '.join(selected_genres)}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Tracks", len(playlist))
            with col2:
                st.metric("Playlist Duration", f"{len(playlist) * 3} mins")  # Assuming average song length of 3 minutes
            
            for _, track in playlist.iterrows():
                st.markdown(f"""
                <div style="background-color: rgba(255,255,255,0.1); padding: 10px; margin: 5px; border-radius: 5px;">
                    <h3>{track['track_name']}</h3>
                    <p>Artist: {track['artists']}</p>
                    <p>Genre: {track['track_genre']}</p>
                    <p>Popularity: {'🔥' * int(track['popularity']/20)}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display fitness scores
            st.write("Fitness Score Evolution:")
            fitness_df = pd.DataFrame({
                'Generation': range(1, len(recommender.best_fitness_scores) + 1),
                'Best Fitness Score': recommender.best_fitness_scores
            })
            st.line_chart(fitness_df.set_index('Generation')['Best Fitness Score'])
            
            # Share button (simulated)
            if st.button("📤 Share My Playlist"):
                st.balloons()
                st.success("Your playlist has been shared! (This is a simulation)")
        else:
            st.warning("No tracks found for the selected genres. Try selecting different genres.")
    else:
        st.warning("Please select at least one genre.")

# Sidebar with fun facts
st.sidebar.markdown("## 💡 Did You Know?")
fun_facts = [
    "The longest officially released song is 'The Rise and Fall of Bossanova (A 13:23:32 song)' by PC III.",
    "The most expensive musical instrument in the world is a Stradivarius violin, with one sold for $15.9 million.",
    "The 'Happy Birthday' song was finally released to the public domain in 2016."
]
st.sidebar.write(random.choice(fun_facts))

# Add a feature to show how many songs are available per genre
if st.checkbox("Show number of songs available per genre"):
    genre_counts = spotify_data['track_genre'].value_counts()
    st.write("Number of songs available in each genre:")
    for genre in available_genres:
        st.write(f"{genre}: {genre_counts.get(genre, 0)} songs")
        
st.markdown("Made with ❤️ by ian ✨")