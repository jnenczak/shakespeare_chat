# shakespeare_chat
Its small local chatbot that answers questions about William Shakespeare using Wikipedia (RAG) for facts, while sounding more Shakespearean thanks to a light LoRA fine-tune on his plays.

We do two things here:
- We train a small LoRA adapter on `shakespeare.txt`. its all Shakespeare works in plain .txt, its use to change how model sounds not what it says
- Then we use RAG and Wikipedia articale to search information about Shakespeare to anserw question about him

Because of file size limits, I couldn’t include the fine-tuned LoRA weights in the GitHub repo. You can download the full project (including the weights) from:

https://drive.google.com/file/d/1Yv5zJLicTihMKVm-3M46UPzgZa2EihIu/view?usp=sharing

Use chat_shakespeare_rag.py to chat. Examples of questions and answers:

> Who was William Shakespeare?

And so, we find our Will, in time past,
Took leave of Oxford, Cambridge, and St. Andrews.
In fair Verona there he lived, did play,
In streets and alleys, in fair fields and parks,
Wherein his art, well used, and well adorned,
Made us forget, poor mortals, that we live.
We saw him in a mask, a masque, a mas,
A comedy, a tragedy, a pageant show.
In every place, in every action,
He played the player, and his players were
The people, men, women, children, rich, poor, gentle, base,
That looked upon him and beheld his parts.
So much the less we saw ourselves, the greater,
For none could see the man but with the eyes
Of one that could perceive the parts.

> When did Shakespeare die, and where was he buried?

And thus I tell of Stratford town,
Where Shakespeare died and was laid to rest,
Within a month’s bower of the grave he took,
A month when he signed his will and made his mark.

Yet none doth know the way
Of how or why, in his last breath he spoke.
As time hath waned, and men grown old,
The tales we hear are oft of drunken lords,
Of feasting and drinking, that the cause was wrought.

But of the truth, no man doth speak,
Nor doth the tale be told, nor ever shall.

> How many children William Shakespeare had? 

Save that he wedded Anne, a match so fair.
Three children did they make thereof.

> What where names of all of his childrens?

Thee said Anne thy wedded wife,
With whom thou hast a brood of three,
Susanna, then twins, Hamnet and Jud.
So shall thy children thee inherit,
Thy love and trust doth prove thy heir.

> What is known about his education?

And though in Stratford learned he was,
The schools did scarce provide him with a name.
A grammar school it seems, indeed,
With Latin texts and classics, yet not high.
The scholars there did crafty men produce,
Some famed, some not, who knew the rules of prose.
But little doth one find concerning him
His days of study spent, nor what he learned.
Nor which of those who passed before him stood
In fame or learning, save the wits of old.

> Tell me what should i eat for dinner 

And let me see a dish that will be light,
A sauce to please your taste, if you do sit,
A pottage hot, or a sweet savouring pie;
That you may like your dinner ere it fly.

> What is meaning of life?

For answer, I do not ask nor seek,
Yet what the man of wit would seek to teach,
In wits of school and in the courtly place,
It hath not much the common man to chase;
That question, what is meaning of life?
Some think it to be found in things divine,
Others in earthly things, in gold and spice,
And some doth think it to be found in wine.
Some find it in the company they choose,
Some find it in their friends’ society.
Others doth find it in the act of sin,
In revels, drunkenness, and idle chins.
Some doth find it in the study of the word,
To read the Bible or to learn the law;
Some find it in the sight of men’s fair faces,
Others doth find it in the face of Mars.
Some doth find it in the sight of beasts

> What is most important in life? 

Nor wealth nor power, nor the wise man’s wit,
Can ease the hearted soul of sin. Then, let
The common people live in peace and ease,
And spend not all the gold they find with care,
On such devices and pretences fair,
That they themselves in darkness do remain. 
