"""DJ Program. The game loop occurs in the function b_next. It gives the user another chance to DJ."""

from random import randint

__author__ = "730323356"

points: int = 0
player: str = ""
role: str = ""
did_dj: bool = False

current_song: str = ""
songs_a: list[str]
songs_a = ["Running", "Body", "BlueBerry Faygo", "After Party", "Orange Soda", "Shooters"]
songs_b: list[str]
songs_b = ["Element", "Party", "Rockstar", "Hot Boy", "Humble", "Power"]
songs_c: list[str]
songs_c = ["Happy", "Don't Trust Me", "Mr. Brightside", "Africa", "Hey Ya", "Baby Got Back"]
mood: int = 100
react: str = ""

rand_one: int = 0
rand_two: int = 0
rand_three: int = 0

EXCITED_FACE: str = "\U0001F92A"
SAD_FACE: str = "\U0001F622"
UNIMPRESSED_FACE: str = "\U0001F612"
LOVING_FACE: str = "\U0001F60D"

good_songs: int = 0
decent_songs: int = 0
wrong_songs: int = 0


def main() -> None:
    """The entrypoint."""
    greet()
    naming()
    path()
    end()


def naming() -> None:
    """Allows player to choose a name and sets up the scenario."""
    global player
    player = input("What is your DJ name? Write it as \"DJ yourname\": ")
    print(f"\nNice to meet you, {player}! Let's get started.")
    b: str = "\nWhile you're vibing out at a party your friends are hosting, the DJ who was hired by your friends"
    c: str = " rage quits upon learning that his shares of GameStop lost most of their value. You have a little "
    d: str = "experience in this field, so you consider taking over as the DJ in order to keep the party going."
    print(f"{b}{c}{d}")


def path() -> None:
    """Takes user input, and begins a story path based on their selected role."""
    global current_song
    global points
    global role
    global mood
    global did_dj
    za: str = "\nYou can: A - take over and use the turntable; B - call someone else to DJ; C -"
    role = input(f"{za} leave the party. \nWhat do you do? Enter the letter of your corresponding choice: ")
    if(role == "A" or role == "a"):
        points += 10
        did_dj = True
        print("\nYou step up to the turntable. The song currently playing is \"Yeah!\" by Usher.")
        song_sequence()
    else:
        if(role == "B" or role == "b"):
            points += 5
            print("The party is back on! For now...")
            b_next(points)
        else:
            if(role == "C" or role == "c"):
                points -= 10
                print("Your friends are disappointed, but you no longer have to worry about a lackluster party.")


def b_next(a: int) -> None:
    """Loops back to original choice of whether to DJ."""
    global points
    global player
    print("A few minutes later, the new DJ learns that his friend got busted for selling vapes to kids.")
    print("He rage quits and storms off. You have another choice to make.")
    print("A friend let it slip that you know how to DJ, and now other people want you to do it.")
    print(f"{player}, you have a social rating of {a}. This choice will change that, for better or worse.")
    feelings: str = input("Are you excited, or afraid? Choose E for excited and A for afraid. ")
    if(feelings == "E" or feelings == "e"):
        a += 10
    else:
        if(feelings == "A" or feelings == "a"):
            a -= 10
    zb: str = "People momentarily forget that you wimped out earlier. "
    print(f"{zb}You now have {a} points. Good luck with your next choice.")
    path()


def end() -> None:
    """Finishes game and comments on user's performance."""
    print(f"Thanks for coming to the party! You earned {points} points.")
    global role
    global mood
    global did_dj
    zc: str = "It's not easy being a DJ, but you did it. Based on the crowd's mood, you played "
    zd: str = "that were out of context."
    if(did_dj):
        print(f"{zc}{good_songs} good songs, {decent_songs} decent songs, and {wrong_songs} songs {zd}")
    print(f"Thanks for coming to the party. Your overall social rating was {points}.")


def song_sequence() -> None:
    """Makes the user select songs a certain number of times."""
    global mood
    count: int = 0
    while count < 10 and mood > 0:
        song_picker()
        count += 1
    

def song_picker() -> None:
    """Makes the user enter a song choice."""
    global points
    global songs_a
    global songs_b
    global songs_c
    global mood
    global good_songs
    global decent_songs
    global wrong_songs
    rand_one = randint(0, 5)
    rand_two = randint(0, 5)
    rand_three = randint(0, 5)
    ze: str = "Song choices (choose the letter of the corresponding song): A - "
    choice: str = input(f"{ze}{songs_a[rand_one]}; B - {songs_b[rand_two]}; C - {songs_c[rand_three]} \n ")
    if(mood <= 120 and mood >= 71):
        if(choice == "A" or choice == "a"):
            print(f"{EXCITED_FACE} They like the hot song! You might want to play something similar next.")
            points += 5
            mood -= 15
            good_songs += 1
        else: 
            if(choice == "B" or choice == "b"):
                print(f"{UNIMPRESSED_FACE} Not bad, but it could have been better. Maybe something more recent?")
                points += 3
                mood -= 10
                decent_songs += 1
            else: 
                if(choice == "C" or choice == "c"):
                    zf: str = "They might not be ready for that yet."
                    print(f"{SAD_FACE} The audience looked a little confused, but some people sung. {zf}")
                    points += 5
                    mood += 5
                    wrong_songs += 1
    else:
        if(mood <= 70 and mood >= 36):
            if(choice == "A"):
                print("Not bad, they enjoyed the hype of the recent song, but they're a little tired of hot songs.")
                points += 3
                mood -= 10
                decent_songs += 1
            else: 
                if(choice == "B"):
                    print(f"{EXCITED_FACE} They liked the throwback! Some other throwbacks might work well.")
                    points += 5
                    mood -= 15
                    good_songs += 1
                else: 
                    if(choice == "C"):
                        print(f"{UNIMPRESSED_FACE} A few people sung, but the mood wasn't quite right.")
                        points += 3
                        mood += 5
                        wrong_songs += 1
        else: 
            if(mood <= 35 and mood >= 1):
                if(choice == "A"):
                    zh: str = "but they're a little tired of hot songs."
                    print(f"Not bad, they enjoyed the hype of the recent song, {zh}")
                    points -= 5
                    mood += 5
                    wrong_songs += 1
                else: 
                    if(choice == "B"):
                        print(f"{UNIMPRESSED_FACE} A few people sung, but the mood wasn't quite right.")
                        points += 3
                        mood -= 5
                        decent_songs += 1
                    else: 
                        if(choice == "C"):
                            zg: str = "In short, they loved the classic."
                            print(f"{LOVING_FACE} People danced together while belting out the lyrics. {zg}")
                            points += 8
                            mood -= 10
                            good_songs += 1


def greet() -> None:
    """Greets player and asks for name."""
    a: str = " Choose Your Own Adventure game puts you in the shoes of a DJ."
    print(f"This{a} Keep the crowd engaged by playing the right songs.")


if __name__ == "__main__":
    main()
